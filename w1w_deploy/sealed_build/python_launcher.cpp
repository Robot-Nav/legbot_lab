#include <Python.h>

#include <filesystem>
#include <iostream>
#include <string>

#include "w1w/device_guard.h"

#ifndef W1W_PYTHON_MODULE
#error "W1W_PYTHON_MODULE must be supplied by the sealed build"
#endif

#ifndef W1W_MODULE_SUBDIR
#error "W1W_MODULE_SUBDIR must be supplied by the sealed build"
#endif

namespace {

std::filesystem::path executable_prefix()
{
    std::error_code error;
    const auto executable = std::filesystem::read_symlink("/proc/self/exe", error);
    if (error)
    {
        return {};
    }
    return executable.parent_path().parent_path();
}

int run_python(int argc, char** argv)
{
    const auto prefix = executable_prefix();
    if (prefix.empty())
    {
        std::cerr << "w1w launcher: cannot resolve executable path" << std::endl;
        return 78;
    }
    const auto module_dir = prefix / "lib" / "w1w" / W1W_MODULE_SUBDIR;

    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.parse_argv = 0;
    PyStatus status = PyConfig_SetBytesArgv(&config, argc, argv);
    if (!PyStatus_Exception(status))
    {
        status = Py_InitializeFromConfig(&config);
    }
    if (PyStatus_Exception(status))
    {
        std::cerr << "w1w launcher: Python initialization failed";
        if (status.err_msg != nullptr)
        {
            std::cerr << ": " << status.err_msg;
        }
        std::cerr << std::endl;
        PyConfig_Clear(&config);
        return 78;
    }
    PyConfig_Clear(&config);

    PyObject* sys_path = PySys_GetObject("path");
    PyObject* path = PyUnicode_FromString(module_dir.c_str());
    if (sys_path == nullptr || path == nullptr || PyList_Insert(sys_path, 0, path) != 0)
    {
        Py_XDECREF(path);
        PyErr_Print();
        Py_Finalize();
        return 78;
    }
    Py_DECREF(path);

    PyObject* module = PyImport_ImportModule(W1W_PYTHON_MODULE);
    if (module == nullptr)
    {
        PyErr_Print();
        Py_Finalize();
        return 78;
    }
    PyObject* main_function = PyObject_GetAttrString(module, "main");
    Py_DECREF(module);
    if (main_function == nullptr || !PyCallable_Check(main_function))
    {
        Py_XDECREF(main_function);
        PyErr_SetString(PyExc_RuntimeError, "sealed Python module has no main()" );
        PyErr_Print();
        Py_Finalize();
        return 78;
    }
    PyObject* result = PyObject_CallNoArgs(main_function);
    Py_DECREF(main_function);
    if (result == nullptr)
    {
        PyErr_Print();
        Py_Finalize();
        return 1;
    }
    int exit_code = 0;
    if (result != Py_None)
    {
        const long value = PyLong_AsLong(result);
        if (!PyErr_Occurred())
        {
            exit_code = static_cast<int>(value);
        }
        else
        {
            PyErr_Print();
            exit_code = 1;
        }
    }
    Py_DECREF(result);
    Py_Finalize();
    return exit_code;
}

} // namespace

int main(int argc, char** argv)
{
    if (!w1w::device_guard::enforce(W1W_PYTHON_MODULE))
    {
        return 77;
    }
    if (argc == 2 && std::string(argv[1]) == "--license-check")
    {
        return 0;
    }
    return run_python(argc, argv);
}
