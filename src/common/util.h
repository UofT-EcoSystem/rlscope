//
// Created by jagle on 11/13/2018.
//

#ifndef DNN_TENSORFLOW_CPP_UTIL_H
#define DNN_TENSORFLOW_CPP_UTIL_H

#include <vector>

#include <type_traits>
#include <typeinfo>
#ifndef _MSC_VER
#   include <cxxabi.h>
#endif
#include <memory>
#include <string>
#include <cstdlib>

template <typename T>
class CircularBuffer {
public:
  size_t _size;
  size_t _used;
  size_t _next_idx;
  std::vector<T> _buffer;
  // STL?
  // CircularBuffer() { }
  CircularBuffer(size_t size) :
      _size(size),
      _used(0),
      _next_idx(0) {
    _buffer.resize(_size);
  }
  CircularBuffer(size_t size, T default_value) :
      _size(size),
      _used(0),
      _next_idx(0),
      _buffer(_size, default_value) {
  }
  void Add(T value) {
    _buffer[_next_idx] = value;
    _next_idx = (_next_idx + 1) % _size;
    _used = std::min(_used + 1, _size);
  }
  void Clear() {
    _used = 0;
  }
};

template <typename T>
T Average(CircularBuffer<T> buffer) {
  if (buffer._used == 0) {
    return NAN;
  }
  T sm = 0;
  size_t end_idx = buffer._used;
  size_t start_idx = 0;
  auto len = end_idx - start_idx + 1;
  for (size_t i = start_idx; i < end_idx; i++) {
    sm += buffer._buffer[i];
  }
  return sm/((T)len);
}

// https://stackoverflow.com/questions/7778734/static-assert-a-way-to-dynamically-customize-error-message
template <class T>
std::string
type_name()
{
    typedef typename std::remove_reference<T>::type TR;
    std::unique_ptr<char, void(*)(void*)> own
           (
#ifndef _MSC_VER
                abi::__cxa_demangle(typeid(TR).name(), nullptr,
                                           nullptr, nullptr),
#else
                nullptr,
#endif
                std::free
           );
    std::string r = own != nullptr ? own.get() : typeid(TR).name();
    if (std::is_const<TR>::value)
        r += " const";
    if (std::is_volatile<TR>::value)
        r += " volatile";
    if (std::is_lvalue_reference<T>::value)
        r += "&";
    else if (std::is_rvalue_reference<T>::value)
        r += "&&";
    return r;
}

#endif //DNN_TENSORFLOW_CPP_UTIL_H
