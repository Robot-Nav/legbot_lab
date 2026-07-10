// 文件用途：非阻塞键盘输入封装。在独立线程中读取终端按键，供 FSM 状态切换或调试使用。
#pragma once

#include <string>
#include <vector>
#include <deque>
#include <termios.h>
#include <unistd.h>
#include <thread>


class Keyboard
{
public:
  Keyboard()
  {
    tcgetattr( fileno( stdin ), &_oldSettings );
    _newSettings = _oldSettings;
    _oldSettings.c_lflag |= ( ICANON |  ECHO);
    _newSettings.c_lflag &= (~ICANON & ~ECHO);

    _startKey();

    _thread_running  = true;
    _readThread = std::thread([this] {
      while (_running) {
        _read();
      }
    });
  }

  ~Keyboard()
  {
    _thread_running = false;
    _pauseKey();
  }

  void update()
  {
    if(_key != _last_key)
    {
      on_pressed = _key != "";
      on_released = _key == "";
    }
    else
    {
      on_pressed = false;
      on_released = false;
    }
    
    _last_key = _key;
  }

  // 获取当前按键字符串。
  std::string key() const { return _key; };

  // 暂停后台读取，阻塞获取一行输入（用于交互式提示）。
  std::string getString(std::string slogan)
  {
    // 暂停后台按键读取
    _running = false;
    _pauseKey();

    std::string stringtemp;
    std::cout << slogan << std::endl;
    std::getline(std::cin, stringtemp);

    // 恢复后台按键读取
    _startKey();
    _running = true;

    return stringtemp;
  }

  // 按键事件标志，调用 update() 后有效。
  bool on_pressed = false;
  bool on_released = false;

  private:
  bool _thread_running = false;
  bool _running = false;
  std::thread _readThread;

  void _read()
  {
    if(_running)
    {
      FD_ZERO(&_fd_set);
      FD_SET( fileno(stdin), &_fd_set);

      _tv.tv_sec = 0;
      _tv.tv_usec = 80000;

      if(select(fileno(stdin)+1, &_fd_set, NULL, NULL, &_tv))
      {
        // 读取一个字符到 _c
        int res = read( fileno(stdin), &_c, 1 );

        // 解析按键：转义序列为方向键
        if(_c != '\033') {
          _key = _c;
        }else{
          int m = read(fileno(stdin), &_c, 1);
          if(_c == '[')
          {
            m = read(fileno(stdin), &_c, 1);
            switch (_c)
            {
            case 'A': _key = "up";    break;
            case 'B': _key = "down";  break;
            case 'C': _key = "right"; break;
            case 'D': _key = "left";  break;
            default:  _key = "";      break;
            }
          }
        }
      }else{
        _key = "";
      }
    }
  }

  // 恢复终端默认设置（规范模式 + 回显）。
  void _pauseKey()
  {
    tcsetattr( fileno( stdin ), TCSANOW, &_oldSettings );
    _running = false;
  }

  // 关闭规范模式与回显，实现非阻塞单字符读取。
  void _startKey()
  {
    tcsetattr( fileno( stdin ), TCSANOW, &_newSettings );
    _running = true;
  }

  fd_set _fd_set;
  char _c = '\0';
  std::string _key, _last_key;
  
  termios _oldSettings, _newSettings;
  timeval _tv;
};