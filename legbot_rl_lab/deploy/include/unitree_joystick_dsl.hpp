/**
 * 文件用途：Unitree 手柄 DSL（领域特定语言）解析器。
 *
 * 该 DSL 用于在手柄/摇杆按键之间定义复杂组合条件，支持 AND（+）、OR（|）、NOT（!）
 * 与括号分组。解析器将字符串表达式编译为可调用谓词，供 FSM 状态切换条件使用。
 *
 * 表达式示例：
 * --- 基础 ---
 * - "A"                  # A 键被按住
 * - "A.on_pressed"       # A 键在本周期刚按下（单帧触发）
 * - "A.on_released"      # A 键在本周期刚释放（单帧触发）
 *
 * --- 多键组合 ---
 * - "A+B"                # A 与 B 同时按住
 * - "RB+X.on_pressed"    # RB 按住且 X 刚按下
 *
 * --- 方向键组合 ---
 * - "up+right"           # 上、右同时按下（对角线）
 *
 * --- 长按检测 ---
 * - "LT(2s) + up"        # LT 长按超过 2 秒且上方向键按下
 * - "LT(3s).pressed"     # 显式指定 .pressed，语义同上
 *
 * --- 多条件或 ---
 * - "X|Y"                # X 或 Y 任一按住
 * - "A.on_pressed|B.on_pressed"  # A 或 B 任一刚按下
 *
 * --- 逻辑非 ---
 * - "!A + B"             # A 未按住且 B 按住
 * - "!(A + B)"           # A 与 B 未同时按住
 * - "!LT(1s)"            # LT 未长按 1 秒（未按或按下的时间不足）
 *
 * --- 嵌套分组 ---
 * - "(A + B) | (X + Y)"  # A+B 或 X+Y 任一满足
 * - "!(A + B | X)"       # A+B 或 X 的组合不允许
 *
 * --- 摇杆轴与触发键 ---
 * - "LX + LY"            # 左摇杆任意方向超出阈值
 * - "RX(1s) + B"         # 右摇杆持续超出阈值 1 秒且 B 按住
 *
 * --- 启动/退出动作 ---
 * - "start.on_pressed"   # Start 键刚按下
 * - "back.on_pressed"    # Back 键刚按下
 * - "!start + !back"     # 两者均未按（常用于空闲状态）
 */
#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <cctype>
#include <memory>
#include <algorithm>
#include <yaml-cpp/yaml.h>

#include <unitree/dds_wrapper/common/unitree_joystick.hpp>

namespace unitree::common::dsl {

// ======================== 词法分析 ========================
struct Token {
  enum Kind {
    kIdent, kNumber,
    kPlus, kOr, kNot,
    kLParen, kRParen,
    kDot, kEnd
  } kind;
  std::string text;
};

class Lexer {
 public:
  explicit Lexer(std::string s) : s_(s) {}
  Token Next() {
    SkipWs();
    if (pos_ >= s_.size()) return {Token::kEnd, ""};
    char c = s_[pos_];
    if (std::isalpha(static_cast<unsigned char>(c))) return Ident();
    if (std::isdigit(static_cast<unsigned char>(c))) return Number(); // 数字仅支持 [1-9] 开头
    ++pos_;
    switch (c) {
      case '+': return {Token::kPlus, "+"};
      case '|': return {Token::kOr, "|"};
      case '!': return {Token::kNot, "!"};
      case '(': return {Token::kLParen, "("};
      case ')': return {Token::kRParen, ")"};
      case '.': return {Token::kDot, "."};
      default:  throw std::runtime_error(std::string("Unexpected char: ") + std::string(1, c) + " near pos=" + std::to_string(pos_-1));
    }
  }
  size_t pos() const { return pos_; }

 private:
  void SkipWs() {
    while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) ++pos_;
  }
  Token Ident() {
    size_t start = pos_;
    while (pos_ < s_.size() &&
           (std::isalnum(static_cast<unsigned char>(s_[pos_])) || s_[pos_]=='_'))
      ++pos_;
    return {Token::kIdent, std::string(s_.substr(start, pos_-start))};
  }
  Token Number() {
    size_t start = pos_;
    if (pos_ < s_.size() && s_[pos_] >= '1' && s_[pos_] <= '9') {
      ++pos_;
    } else {
      throw std::runtime_error("Expected a number starting with [1-9] near pos=" + std::to_string(pos_));
    }
    while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) {
      ++pos_;
    }
    return {Token::kNumber, std::string(s_.substr(start, pos_ - start))};
  }

  const std::string s_;
  size_t pos_{0};
};

// ======================== 抽象语法树与语义 ========================
enum class Field { kPressed, kOnPressed, kOnReleased, kHoldTimeGE };

struct Atom {
  std::string name;         // 按键名，例如 LT、RB、up
  Field field{Field::kPressed};
  float hold_seconds{0.f};  // 当 field 为 kHoldTimeGE 时使用，表示长按秒数
};

struct Node {
  enum Kind { kAtom, kNot, kAnd, kOr } kind{kAtom};
  Atom atom;                  // 原子节点时有效
  std::unique_ptr<Node> lhs;  // 非：子节点；与/或：左子树
  std::unique_ptr<Node> rhs;  // 与/或：右子树
};

// 字符串转小写，使按键名大小写不敏感。
inline std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
  return s;
}

// 按按键名从 UnitreeJoystick 中取出对应 KeyBase（大小写不敏感）。
inline const KeyBase& GetKey(const UnitreeJoystick& joy, std::string_view name_sv) {
  const std::string name = ToLower(std::string{name_sv});
  static const std::unordered_map<std::string, const KeyBase* (*)(const UnitreeJoystick&)> kMap = {
    {"back", [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.back); }},
    {"start",[](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.start); }},
    {"ls",   [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.LS); }},
    {"rs",   [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.RS); }},
    {"lb",   [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.LB); }},
    {"rb",   [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.RB); }},
    {"a",    [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.A); }},
    {"b",    [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.B); }},
    {"x",    [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.X); }},
    {"y",    [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.Y); }},
    {"up",   [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.up); }},
    {"down", [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.down); }},
    {"left", [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.left); }},
    {"right",[](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.right); }},
    {"f1",   [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.F1); }},
    {"f2",   [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.F2); }},
    {"lx",   [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.lx); }},
    {"ly",   [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.ly); }},
    {"rx",   [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.rx); }},
    {"ry",   [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.ry); }},
    {"lt",   [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.LT); }},
    {"rt",   [](auto& j)->const KeyBase*{ return &static_cast<const KeyBase&>(j.RT); }},
  };
  auto it = kMap.find(name);
  if (it == kMap.end()) throw std::runtime_error("Unknown key name: " + std::string(name_sv));
  return *it->second(joy);
}

// ======================== 递归下降解析器 ========================
// 支持：! 一元非、+ 逻辑与、| 逻辑或、() 分组。
// 原子语法：name [ '(' number ['s'|'sec'|'secs'] ')' ] [ '.' (pressed|on_pressed|on_released) ]
class Parser {
 public:
  explicit Parser(std::string expr) : lex_(expr) { 
    tok_ = lex_.Next(); 
  }
  std::unique_ptr<Node> Parse() {
    auto n = ParseOr();
    if (tok_.kind != Token::kEnd) {
      throw std::runtime_error("Trailing tokens near pos=" + std::to_string(lex_.pos()));
    }
    return n;
  }

 private:
  std::unique_ptr<Node> ParseOr() {
    auto left = ParseAnd();
    while (tok_.kind == Token::kOr) {
      Eat(Token::kOr);
      auto right = ParseAnd();
      auto n = std::make_unique<Node>();
      n->kind = Node::kOr;
      n->lhs = std::move(left);
      n->rhs = std::move(right);
      left = std::move(n);
    }
    return left;
  }
  std::unique_ptr<Node> ParseAnd() {
    auto left = ParseUnary();
    while (tok_.kind == Token::kPlus) {
      Eat(Token::kPlus);
      auto right = ParseUnary();
      auto n = std::make_unique<Node>();
      n->kind = Node::kAnd;
      n->lhs = std::move(left);
      n->rhs = std::move(right);
      left = std::move(n);
    }
    return left;
  }
  std::unique_ptr<Node> ParseUnary() {
    if (tok_.kind == Token::kNot) {
      Eat(Token::kNot);
      auto child = ParseUnary();
      auto n = std::make_unique<Node>();
      n->kind = Node::kNot;
      n->lhs = std::move(child);
      return n;
    }
    if (tok_.kind == Token::kLParen) {
      Eat(Token::kLParen);
      auto inside = ParseOr();
      Eat(Token::kRParen);
      return inside;
    }
    return ParseAtom();
  }

  std::unique_ptr<Node> ParseAtom() {
    if (tok_.kind != Token::kIdent) throw std::runtime_error("Expected identifier near pos=" + std::to_string(lex_.pos()));
    Atom a;
    a.name = tok_.text;
    Eat(Token::kIdent);

    // 可选长按时长：name '(' number ['s'|'sec'|'secs'] ')'
    if (tok_.kind == Token::kLParen) {
      Eat(Token::kLParen);
      if (tok_.kind != Token::kNumber) throw std::runtime_error("Expected hold seconds number near pos=" + std::to_string(lex_.pos()));
      a.hold_seconds = std::stof(tok_.text);
      Eat(Token::kNumber);
      // 可选时间单位
      if (tok_.kind == Token::kIdent) {
        std::string unit = ToLower(tok_.text);
        if (unit == "s" || unit == "sec" || unit == "secs") {
          Eat(Token::kIdent);
        } else {
          // 对未知或缺失单位给出明确报错
          throw std::runtime_error("Unknown time unit '" + tok_.text + "'; use 's'/'sec'");
        }
      }
      Eat(Token::kRParen);
      a.field = Field::kHoldTimeGE;
    }

    // 可选显式状态
    if (tok_.kind == Token::kDot) {
      Eat(Token::kDot);
      if (tok_.kind != Token::kIdent) throw std::runtime_error("Expected state after '.' near pos=" + std::to_string(lex_.pos()));
      const std::string st = tok_.text;
      if (st == "on_pressed")      a.field = Field::kOnPressed;
      else if (st == "on_released") a.field = Field::kOnReleased;
      else if (st == "pressed")     a.field = Field::kPressed;
      else throw std::runtime_error("Unknown field: " + st + " (allowed: pressed|on_pressed|on_released)");
      Eat(Token::kIdent);
    }

    auto n = std::make_unique<Node>();
    n->kind = Node::kAtom;
    n->atom = a;

    return n;
  }

  void Eat(Token::Kind k) {
    if (tok_.kind != k) {
      throw std::runtime_error("Unexpected token near pos=" + std::to_string(lex_.pos()));
    }
    tok_ = lex_.Next();
  }

  Lexer lex_;
  Token tok_{Token::kEnd, ""};
};

// ======================== 编译为可执行谓词 ========================
inline std::function<bool(const UnitreeJoystick&)> Compile(const Node& n) {
  switch (n.kind) {
    case Node::kAtom: {
      Atom a = n.atom;
      return [a](const UnitreeJoystick& joy) -> bool {
        const KeyBase& kb = GetKey(joy, a.name);
        switch (a.field) {
          case Field::kPressed:     return kb.pressed;
          case Field::kOnPressed:   return kb.on_pressed;
          case Field::kOnReleased:  return kb.on_released;
          case Field::kHoldTimeGE:  return kb.pressed && (kb.pressed_time >= a.hold_seconds);
        }
        return false;
      };
    }
    case Node::kNot: {
      auto child = Compile(*n.lhs);
      return [child](const UnitreeJoystick& joy){ return !child(joy); };
    }
    case Node::kAnd: {
      auto l = Compile(*n.lhs);
      auto r = Compile(*n.rhs);
      return [l, r](const UnitreeJoystick& joy){ return l(joy) && r(joy); };
    }
    case Node::kOr: {
      auto l = Compile(*n.lhs);
      auto r = Compile(*n.rhs);
      return [l, r](const UnitreeJoystick& joy){ return l(joy) || r(joy); };
    }
  }
  throw std::runtime_error("Invalid node kind");
}

} // namespace unitree::common::dsl