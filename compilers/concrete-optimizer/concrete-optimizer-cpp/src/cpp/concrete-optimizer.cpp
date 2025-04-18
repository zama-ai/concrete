#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace rust {
inline namespace cxxbridge1 {
// #include "rust/cxx.h"

#ifndef CXXBRIDGE1_PANIC
#define CXXBRIDGE1_PANIC
template <typename Exception>
void panic [[noreturn]] (const char *msg);
#endif // CXXBRIDGE1_PANIC

struct unsafe_bitcopy_t;

namespace {
template <typename T>
class impl;
} // namespace

template <typename T>
::std::size_t size_of();
template <typename T>
::std::size_t align_of();

#ifndef CXXBRIDGE1_RUST_STRING
#define CXXBRIDGE1_RUST_STRING
class String final {
public:
  String() noexcept;
  String(const String &) noexcept;
  String(String &&) noexcept;
  ~String() noexcept;

  String(const std::string &);
  String(const char *);
  String(const char *, std::size_t);
  String(const char16_t *);
  String(const char16_t *, std::size_t);

  static String lossy(const std::string &) noexcept;
  static String lossy(const char *) noexcept;
  static String lossy(const char *, std::size_t) noexcept;
  static String lossy(const char16_t *) noexcept;
  static String lossy(const char16_t *, std::size_t) noexcept;

  String &operator=(const String &) &noexcept;
  String &operator=(String &&) &noexcept;

  explicit operator std::string() const;

  const char *data() const noexcept;
  std::size_t size() const noexcept;
  std::size_t length() const noexcept;
  bool empty() const noexcept;

  const char *c_str() noexcept;

  std::size_t capacity() const noexcept;
  void reserve(size_t new_cap) noexcept;

  using iterator = char *;
  iterator begin() noexcept;
  iterator end() noexcept;

  using const_iterator = const char *;
  const_iterator begin() const noexcept;
  const_iterator end() const noexcept;
  const_iterator cbegin() const noexcept;
  const_iterator cend() const noexcept;

  bool operator==(const String &) const noexcept;
  bool operator!=(const String &) const noexcept;
  bool operator<(const String &) const noexcept;
  bool operator<=(const String &) const noexcept;
  bool operator>(const String &) const noexcept;
  bool operator>=(const String &) const noexcept;

  void swap(String &) noexcept;

  String(unsafe_bitcopy_t, const String &) noexcept;

private:
  struct lossy_t;
  String(lossy_t, const char *, std::size_t) noexcept;
  String(lossy_t, const char16_t *, std::size_t) noexcept;
  friend void swap(String &lhs, String &rhs) noexcept { lhs.swap(rhs); }

  std::array<std::uintptr_t, 3> repr;
};
#endif // CXXBRIDGE1_RUST_STRING

#ifndef CXXBRIDGE1_RUST_STR
#define CXXBRIDGE1_RUST_STR
class Str final {
public:
  Str() noexcept;
  Str(const String &) noexcept;
  Str(const std::string &);
  Str(const char *);
  Str(const char *, std::size_t);

  Str &operator=(const Str &) &noexcept = default;

  explicit operator std::string() const;

  const char *data() const noexcept;
  std::size_t size() const noexcept;
  std::size_t length() const noexcept;
  bool empty() const noexcept;

  Str(const Str &) noexcept = default;
  ~Str() noexcept = default;

  using iterator = const char *;
  using const_iterator = const char *;
  const_iterator begin() const noexcept;
  const_iterator end() const noexcept;
  const_iterator cbegin() const noexcept;
  const_iterator cend() const noexcept;

  bool operator==(const Str &) const noexcept;
  bool operator!=(const Str &) const noexcept;
  bool operator<(const Str &) const noexcept;
  bool operator<=(const Str &) const noexcept;
  bool operator>(const Str &) const noexcept;
  bool operator>=(const Str &) const noexcept;

  void swap(Str &) noexcept;

private:
  class uninit;
  Str(uninit) noexcept;
  friend impl<Str>;

  std::array<std::uintptr_t, 2> repr;
};
#endif // CXXBRIDGE1_RUST_STR

#ifndef CXXBRIDGE1_RUST_SLICE
#define CXXBRIDGE1_RUST_SLICE
namespace detail {
template <bool>
struct copy_assignable_if {};

template <>
struct copy_assignable_if<false> {
  copy_assignable_if() noexcept = default;
  copy_assignable_if(const copy_assignable_if &) noexcept = default;
  copy_assignable_if &operator=(const copy_assignable_if &) &noexcept = delete;
  copy_assignable_if &operator=(copy_assignable_if &&) &noexcept = default;
};
} // namespace detail

template <typename T>
class Slice final
    : private detail::copy_assignable_if<std::is_const<T>::value> {
public:
  using value_type = T;

  Slice() noexcept;
  Slice(T *, std::size_t count) noexcept;

  template <typename C>
  explicit Slice(C& c) : Slice(c.data(), c.size()) {}

  Slice &operator=(const Slice<T> &) &noexcept = default;
  Slice &operator=(Slice<T> &&) &noexcept = default;

  T *data() const noexcept;
  std::size_t size() const noexcept;
  std::size_t length() const noexcept;
  bool empty() const noexcept;

  T &operator[](std::size_t n) const noexcept;
  T &at(std::size_t n) const;
  T &front() const noexcept;
  T &back() const noexcept;

  Slice(const Slice<T> &) noexcept = default;
  ~Slice() noexcept = default;

  class iterator;
  iterator begin() const noexcept;
  iterator end() const noexcept;

  void swap(Slice &) noexcept;

private:
  class uninit;
  Slice(uninit) noexcept;
  friend impl<Slice>;
  friend void sliceInit(void *, const void *, std::size_t) noexcept;
  friend void *slicePtr(const void *) noexcept;
  friend std::size_t sliceLen(const void *) noexcept;

  std::array<std::uintptr_t, 2> repr;
};

template <typename T>
class Slice<T>::iterator final {
public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = typename std::add_pointer<T>::type;
  using reference = typename std::add_lvalue_reference<T>::type;

  reference operator*() const noexcept;
  pointer operator->() const noexcept;
  reference operator[](difference_type) const noexcept;

  iterator &operator++() noexcept;
  iterator operator++(int) noexcept;
  iterator &operator--() noexcept;
  iterator operator--(int) noexcept;

  iterator &operator+=(difference_type) noexcept;
  iterator &operator-=(difference_type) noexcept;
  iterator operator+(difference_type) const noexcept;
  iterator operator-(difference_type) const noexcept;
  difference_type operator-(const iterator &) const noexcept;

  bool operator==(const iterator &) const noexcept;
  bool operator!=(const iterator &) const noexcept;
  bool operator<(const iterator &) const noexcept;
  bool operator<=(const iterator &) const noexcept;
  bool operator>(const iterator &) const noexcept;
  bool operator>=(const iterator &) const noexcept;

private:
  friend class Slice;
  void *pos;
  std::size_t stride;
};

template <typename T>
Slice<T>::Slice() noexcept {
  sliceInit(this, reinterpret_cast<void *>(align_of<T>()), 0);
}

template <typename T>
Slice<T>::Slice(T *s, std::size_t count) noexcept {
  assert(s != nullptr || count == 0);
  sliceInit(this,
            s == nullptr && count == 0
                ? reinterpret_cast<void *>(align_of<T>())
                : const_cast<typename std::remove_const<T>::type *>(s),
            count);
}

template <typename T>
T *Slice<T>::data() const noexcept {
  return reinterpret_cast<T *>(slicePtr(this));
}

template <typename T>
std::size_t Slice<T>::size() const noexcept {
  return sliceLen(this);
}

template <typename T>
std::size_t Slice<T>::length() const noexcept {
  return this->size();
}

template <typename T>
bool Slice<T>::empty() const noexcept {
  return this->size() == 0;
}

template <typename T>
T &Slice<T>::operator[](std::size_t n) const noexcept {
  assert(n < this->size());
  auto ptr = static_cast<char *>(slicePtr(this)) + size_of<T>() * n;
  return *reinterpret_cast<T *>(ptr);
}

template <typename T>
T &Slice<T>::at(std::size_t n) const {
  if (n >= this->size()) {
    panic<std::out_of_range>("rust::Slice index out of range");
  }
  return (*this)[n];
}

template <typename T>
T &Slice<T>::front() const noexcept {
  assert(!this->empty());
  return (*this)[0];
}

template <typename T>
T &Slice<T>::back() const noexcept {
  assert(!this->empty());
  return (*this)[this->size() - 1];
}

template <typename T>
typename Slice<T>::iterator::reference
Slice<T>::iterator::operator*() const noexcept {
  return *static_cast<T *>(this->pos);
}

template <typename T>
typename Slice<T>::iterator::pointer
Slice<T>::iterator::operator->() const noexcept {
  return static_cast<T *>(this->pos);
}

template <typename T>
typename Slice<T>::iterator::reference Slice<T>::iterator::operator[](
    typename Slice<T>::iterator::difference_type n) const noexcept {
  auto ptr = static_cast<char *>(this->pos) + this->stride * n;
  return *reinterpret_cast<T *>(ptr);
}

template <typename T>
typename Slice<T>::iterator &Slice<T>::iterator::operator++() noexcept {
  this->pos = static_cast<char *>(this->pos) + this->stride;
  return *this;
}

template <typename T>
typename Slice<T>::iterator Slice<T>::iterator::operator++(int) noexcept {
  auto ret = iterator(*this);
  this->pos = static_cast<char *>(this->pos) + this->stride;
  return ret;
}

template <typename T>
typename Slice<T>::iterator &Slice<T>::iterator::operator--() noexcept {
  this->pos = static_cast<char *>(this->pos) - this->stride;
  return *this;
}

template <typename T>
typename Slice<T>::iterator Slice<T>::iterator::operator--(int) noexcept {
  auto ret = iterator(*this);
  this->pos = static_cast<char *>(this->pos) - this->stride;
  return ret;
}

template <typename T>
typename Slice<T>::iterator &Slice<T>::iterator::operator+=(
    typename Slice<T>::iterator::difference_type n) noexcept {
  this->pos = static_cast<char *>(this->pos) + this->stride * n;
  return *this;
}

template <typename T>
typename Slice<T>::iterator &Slice<T>::iterator::operator-=(
    typename Slice<T>::iterator::difference_type n) noexcept {
  this->pos = static_cast<char *>(this->pos) - this->stride * n;
  return *this;
}

template <typename T>
typename Slice<T>::iterator Slice<T>::iterator::operator+(
    typename Slice<T>::iterator::difference_type n) const noexcept {
  auto ret = iterator(*this);
  ret.pos = static_cast<char *>(this->pos) + this->stride * n;
  return ret;
}

template <typename T>
typename Slice<T>::iterator Slice<T>::iterator::operator-(
    typename Slice<T>::iterator::difference_type n) const noexcept {
  auto ret = iterator(*this);
  ret.pos = static_cast<char *>(this->pos) - this->stride * n;
  return ret;
}

template <typename T>
typename Slice<T>::iterator::difference_type
Slice<T>::iterator::operator-(const iterator &other) const noexcept {
  auto diff = std::distance(static_cast<char *>(other.pos),
                            static_cast<char *>(this->pos));
  return diff / static_cast<typename Slice<T>::iterator::difference_type>(
                    this->stride);
}

template <typename T>
bool Slice<T>::iterator::operator==(const iterator &other) const noexcept {
  return this->pos == other.pos;
}

template <typename T>
bool Slice<T>::iterator::operator!=(const iterator &other) const noexcept {
  return this->pos != other.pos;
}

template <typename T>
bool Slice<T>::iterator::operator<(const iterator &other) const noexcept {
  return this->pos < other.pos;
}

template <typename T>
bool Slice<T>::iterator::operator<=(const iterator &other) const noexcept {
  return this->pos <= other.pos;
}

template <typename T>
bool Slice<T>::iterator::operator>(const iterator &other) const noexcept {
  return this->pos > other.pos;
}

template <typename T>
bool Slice<T>::iterator::operator>=(const iterator &other) const noexcept {
  return this->pos >= other.pos;
}

template <typename T>
typename Slice<T>::iterator Slice<T>::begin() const noexcept {
  iterator it;
  it.pos = slicePtr(this);
  it.stride = size_of<T>();
  return it;
}

template <typename T>
typename Slice<T>::iterator Slice<T>::end() const noexcept {
  iterator it = this->begin();
  it.pos = static_cast<char *>(it.pos) + it.stride * this->size();
  return it;
}

template <typename T>
void Slice<T>::swap(Slice &rhs) noexcept {
  std::swap(*this, rhs);
}
#endif // CXXBRIDGE1_RUST_SLICE

#ifndef CXXBRIDGE1_RUST_BOX
#define CXXBRIDGE1_RUST_BOX
template <typename T>
class Box final {
public:
  using element_type = T;
  using const_pointer =
      typename std::add_pointer<typename std::add_const<T>::type>::type;
  using pointer = typename std::add_pointer<T>::type;

  Box() = delete;
  Box(Box &&) noexcept;
  ~Box() noexcept;

  explicit Box(const T &);
  explicit Box(T &&);

  Box &operator=(Box &&) &noexcept;

  const T *operator->() const noexcept;
  const T &operator*() const noexcept;
  T *operator->() noexcept;
  T &operator*() noexcept;

  template <typename... Fields>
  static Box in_place(Fields &&...);

  void swap(Box &) noexcept;

  static Box from_raw(T *) noexcept;

  T *into_raw() noexcept;

  /* Deprecated */ using value_type = element_type;

private:
  class uninit;
  class allocation;
  Box(uninit) noexcept;
  void drop() noexcept;

  friend void swap(Box &lhs, Box &rhs) noexcept { lhs.swap(rhs); }

  T *ptr;
};

template <typename T>
class Box<T>::uninit {};

template <typename T>
class Box<T>::allocation {
  static T *alloc() noexcept;
  static void dealloc(T *) noexcept;

public:
  allocation() noexcept : ptr(alloc()) {}
  ~allocation() noexcept {
    if (this->ptr) {
      dealloc(this->ptr);
    }
  }
  T *ptr;
};

template <typename T>
Box<T>::Box(Box &&other) noexcept : ptr(other.ptr) {
  other.ptr = nullptr;
}

template <typename T>
Box<T>::Box(const T &val) {
  allocation alloc;
  ::new (alloc.ptr) T(val);
  this->ptr = alloc.ptr;
  alloc.ptr = nullptr;
}

template <typename T>
Box<T>::Box(T &&val) {
  allocation alloc;
  ::new (alloc.ptr) T(std::move(val));
  this->ptr = alloc.ptr;
  alloc.ptr = nullptr;
}

template <typename T>
Box<T>::~Box() noexcept {
  if (this->ptr) {
    this->drop();
  }
}

template <typename T>
Box<T> &Box<T>::operator=(Box &&other) &noexcept {
  if (this->ptr) {
    this->drop();
  }
  this->ptr = other.ptr;
  other.ptr = nullptr;
  return *this;
}

template <typename T>
const T *Box<T>::operator->() const noexcept {
  return this->ptr;
}

template <typename T>
const T &Box<T>::operator*() const noexcept {
  return *this->ptr;
}

template <typename T>
T *Box<T>::operator->() noexcept {
  return this->ptr;
}

template <typename T>
T &Box<T>::operator*() noexcept {
  return *this->ptr;
}

template <typename T>
template <typename... Fields>
Box<T> Box<T>::in_place(Fields &&...fields) {
  allocation alloc;
  auto ptr = alloc.ptr;
  ::new (ptr) T{std::forward<Fields>(fields)...};
  alloc.ptr = nullptr;
  return from_raw(ptr);
}

template <typename T>
void Box<T>::swap(Box &rhs) noexcept {
  using std::swap;
  swap(this->ptr, rhs.ptr);
}

template <typename T>
Box<T> Box<T>::from_raw(T *raw) noexcept {
  Box box = uninit{};
  box.ptr = raw;
  return box;
}

template <typename T>
T *Box<T>::into_raw() noexcept {
  T *raw = this->ptr;
  this->ptr = nullptr;
  return raw;
}

template <typename T>
Box<T>::Box(uninit) noexcept {}
#endif // CXXBRIDGE1_RUST_BOX

#ifndef CXXBRIDGE1_RUST_BITCOPY_T
#define CXXBRIDGE1_RUST_BITCOPY_T
struct unsafe_bitcopy_t final {
  explicit unsafe_bitcopy_t() = default;
};
#endif // CXXBRIDGE1_RUST_BITCOPY_T

#ifndef CXXBRIDGE1_RUST_VEC
#define CXXBRIDGE1_RUST_VEC
template <typename T>
class Vec final {
public:
  using value_type = T;

  Vec() noexcept;
  Vec(std::initializer_list<T>);
  Vec(const Vec &);
  Vec(Vec &&) noexcept;
  ~Vec() noexcept;

  Vec &operator=(Vec &&) &noexcept;
  Vec &operator=(const Vec &) &;

  std::size_t size() const noexcept;
  bool empty() const noexcept;
  const T *data() const noexcept;
  T *data() noexcept;
  std::size_t capacity() const noexcept;

  const T &operator[](std::size_t n) const noexcept;
  const T &at(std::size_t n) const;
  const T &front() const noexcept;
  const T &back() const noexcept;

  T &operator[](std::size_t n) noexcept;
  T &at(std::size_t n);
  T &front() noexcept;
  T &back() noexcept;

  void reserve(std::size_t new_cap);
  void push_back(const T &value);
  void push_back(T &&value);
  template <typename... Args>
  void emplace_back(Args &&...args);
  void truncate(std::size_t len);
  void clear();

  using iterator = typename Slice<T>::iterator;
  iterator begin() noexcept;
  iterator end() noexcept;

  using const_iterator = typename Slice<const T>::iterator;
  const_iterator begin() const noexcept;
  const_iterator end() const noexcept;
  const_iterator cbegin() const noexcept;
  const_iterator cend() const noexcept;

  void swap(Vec &) noexcept;

  Vec(unsafe_bitcopy_t, const Vec &) noexcept;

private:
  void reserve_total(std::size_t new_cap) noexcept;
  void set_len(std::size_t len) noexcept;
  void drop() noexcept;

  friend void swap(Vec &lhs, Vec &rhs) noexcept { lhs.swap(rhs); }

  std::array<std::uintptr_t, 3> repr;
};

template <typename T>
Vec<T>::Vec(std::initializer_list<T> init) : Vec{} {
  this->reserve_total(init.size());
  std::move(init.begin(), init.end(), std::back_inserter(*this));
}

template <typename T>
Vec<T>::Vec(const Vec &other) : Vec() {
  this->reserve_total(other.size());
  std::copy(other.begin(), other.end(), std::back_inserter(*this));
}

template <typename T>
Vec<T>::Vec(Vec &&other) noexcept : repr(other.repr) {
  new (&other) Vec();
}

template <typename T>
Vec<T>::~Vec() noexcept {
  this->drop();
}

template <typename T>
Vec<T> &Vec<T>::operator=(Vec &&other) &noexcept {
  this->drop();
  this->repr = other.repr;
  new (&other) Vec();
  return *this;
}

template <typename T>
Vec<T> &Vec<T>::operator=(const Vec &other) & {
  if (this != &other) {
    this->drop();
    new (this) Vec(other);
  }
  return *this;
}

template <typename T>
bool Vec<T>::empty() const noexcept {
  return this->size() == 0;
}

template <typename T>
T *Vec<T>::data() noexcept {
  return const_cast<T *>(const_cast<const Vec<T> *>(this)->data());
}

template <typename T>
const T &Vec<T>::operator[](std::size_t n) const noexcept {
  assert(n < this->size());
  auto data = reinterpret_cast<const char *>(this->data());
  return *reinterpret_cast<const T *>(data + n * size_of<T>());
}

template <typename T>
const T &Vec<T>::at(std::size_t n) const {
  if (n >= this->size()) {
    panic<std::out_of_range>("rust::Vec index out of range");
  }
  return (*this)[n];
}

template <typename T>
const T &Vec<T>::front() const noexcept {
  assert(!this->empty());
  return (*this)[0];
}

template <typename T>
const T &Vec<T>::back() const noexcept {
  assert(!this->empty());
  return (*this)[this->size() - 1];
}

template <typename T>
T &Vec<T>::operator[](std::size_t n) noexcept {
  assert(n < this->size());
  auto data = reinterpret_cast<char *>(this->data());
  return *reinterpret_cast<T *>(data + n * size_of<T>());
}

template <typename T>
T &Vec<T>::at(std::size_t n) {
  if (n >= this->size()) {
    panic<std::out_of_range>("rust::Vec index out of range");
  }
  return (*this)[n];
}

template <typename T>
T &Vec<T>::front() noexcept {
  assert(!this->empty());
  return (*this)[0];
}

template <typename T>
T &Vec<T>::back() noexcept {
  assert(!this->empty());
  return (*this)[this->size() - 1];
}

template <typename T>
void Vec<T>::reserve(std::size_t new_cap) {
  this->reserve_total(new_cap);
}

template <typename T>
void Vec<T>::push_back(const T &value) {
  this->emplace_back(value);
}

template <typename T>
void Vec<T>::push_back(T &&value) {
  this->emplace_back(std::move(value));
}

template <typename T>
template <typename... Args>
void Vec<T>::emplace_back(Args &&...args) {
  auto size = this->size();
  this->reserve_total(size + 1);
  ::new (reinterpret_cast<T *>(reinterpret_cast<char *>(this->data()) +
                               size * size_of<T>()))
      T(std::forward<Args>(args)...);
  this->set_len(size + 1);
}

template <typename T>
void Vec<T>::clear() {
  this->truncate(0);
}

template <typename T>
typename Vec<T>::iterator Vec<T>::begin() noexcept {
  return Slice<T>(this->data(), this->size()).begin();
}

template <typename T>
typename Vec<T>::iterator Vec<T>::end() noexcept {
  return Slice<T>(this->data(), this->size()).end();
}

template <typename T>
typename Vec<T>::const_iterator Vec<T>::begin() const noexcept {
  return this->cbegin();
}

template <typename T>
typename Vec<T>::const_iterator Vec<T>::end() const noexcept {
  return this->cend();
}

template <typename T>
typename Vec<T>::const_iterator Vec<T>::cbegin() const noexcept {
  return Slice<const T>(this->data(), this->size()).begin();
}

template <typename T>
typename Vec<T>::const_iterator Vec<T>::cend() const noexcept {
  return Slice<const T>(this->data(), this->size()).end();
}

template <typename T>
void Vec<T>::swap(Vec &rhs) noexcept {
  using std::swap;
  swap(this->repr, rhs.repr);
}

template <typename T>
Vec<T>::Vec(unsafe_bitcopy_t, const Vec &bits) noexcept : repr(bits.repr) {}
#endif // CXXBRIDGE1_RUST_VEC

#ifndef CXXBRIDGE1_RUST_OPAQUE
#define CXXBRIDGE1_RUST_OPAQUE
class Opaque {
public:
  Opaque() = delete;
  Opaque(const Opaque &) = delete;
  ~Opaque() = delete;
};
#endif // CXXBRIDGE1_RUST_OPAQUE

#ifndef CXXBRIDGE1_IS_COMPLETE
#define CXXBRIDGE1_IS_COMPLETE
namespace detail {
namespace {
template <typename T, typename = std::size_t>
struct is_complete : std::false_type {};
template <typename T>
struct is_complete<T, decltype(sizeof(T))> : std::true_type {};
} // namespace
} // namespace detail
#endif // CXXBRIDGE1_IS_COMPLETE

#ifndef CXXBRIDGE1_LAYOUT
#define CXXBRIDGE1_LAYOUT
class layout {
  template <typename T>
  friend std::size_t size_of();
  template <typename T>
  friend std::size_t align_of();
  template <typename T>
  static typename std::enable_if<std::is_base_of<Opaque, T>::value,
                                 std::size_t>::type
  do_size_of() {
    return T::layout::size();
  }
  template <typename T>
  static typename std::enable_if<!std::is_base_of<Opaque, T>::value,
                                 std::size_t>::type
  do_size_of() {
    return sizeof(T);
  }
  template <typename T>
  static
      typename std::enable_if<detail::is_complete<T>::value, std::size_t>::type
      size_of() {
    return do_size_of<T>();
  }
  template <typename T>
  static typename std::enable_if<std::is_base_of<Opaque, T>::value,
                                 std::size_t>::type
  do_align_of() {
    return T::layout::align();
  }
  template <typename T>
  static typename std::enable_if<!std::is_base_of<Opaque, T>::value,
                                 std::size_t>::type
  do_align_of() {
    return alignof(T);
  }
  template <typename T>
  static
      typename std::enable_if<detail::is_complete<T>::value, std::size_t>::type
      align_of() {
    return do_align_of<T>();
  }
};

template <typename T>
std::size_t size_of() {
  return layout::size_of<T>();
}

template <typename T>
std::size_t align_of() {
  return layout::align_of<T>();
}
#endif // CXXBRIDGE1_LAYOUT

namespace detail {
template <typename T, typename = void *>
struct operator_new {
  void *operator()(::std::size_t sz) { return ::operator new(sz); }
};

template <typename T>
struct operator_new<T, decltype(T::operator new(sizeof(T)))> {
  void *operator()(::std::size_t sz) { return T::operator new(sz); }
};
} // namespace detail

template <typename T>
union ManuallyDrop {
  T value;
  ManuallyDrop(T &&value) : value(::std::move(value)) {}
  ~ManuallyDrop() {}
};

template <typename T>
union MaybeUninit {
  T value;
  void *operator new(::std::size_t sz) { return detail::operator_new<T>{}(sz); }
  MaybeUninit() {}
  ~MaybeUninit() {}
};
} // namespace cxxbridge1
} // namespace rust

struct PrivateFunctionalPackingBoostrapKey;
struct CircuitKeys;
namespace concrete_optimizer {
  struct Dag;
  struct DagBuilder;
  struct Location;
  struct ExternalPartition;
  struct Weights;
  enum class Encoding : ::std::uint8_t;
  enum class MultiParamStrategy : ::std::uint8_t;
  struct Options;
  namespace dag {
    struct OperatorIndex;
    struct DagSolution;
    struct BrDecompositionParameters;
    struct KsDecompositionParameters;
    struct SecretLweKey;
    struct BootstrapKey;
    struct KeySwitchKey;
    struct ConversionKeySwitchKey;
    struct CircuitBoostrapKey;
    struct InstructionKeys;
    struct CircuitSolution;
  }
  namespace v0 {
    struct Solution;
  }
  namespace restriction {
    struct RangeRestriction;
    struct LweSecretKeyInfo;
    struct LweBootstrapKeyInfo;
    struct LweKeyswitchKeyInfo;
    struct KeysetInfo;
    struct KeysetRestriction;
  }
  namespace utils {
    struct PartitionDefinition;
  }
}

namespace concrete_optimizer {
#ifndef CXXBRIDGE1_STRUCT_concrete_optimizer$Dag
#define CXXBRIDGE1_STRUCT_concrete_optimizer$Dag
struct Dag final : public ::rust::Opaque {
  ::rust::Box<::concrete_optimizer::DagBuilder> builder(::rust::String circuit) noexcept;
  ::rust::String dump() const noexcept;
  ::concrete_optimizer::dag::DagSolution optimize(::concrete_optimizer::Options const &options) const noexcept;
  void add_composition(::std::string const &from_func, ::std::size_t from_pos, ::std::string const &to_func, ::std::size_t to_pos) noexcept;
  void add_all_compositions() noexcept;
  ::std::size_t get_circuit_count() const noexcept;
  ::concrete_optimizer::dag::CircuitSolution optimize_multi(::concrete_optimizer::Options const &options) const noexcept;
  ::rust::Vec<::concrete_optimizer::dag::OperatorIndex> get_input_indices() const noexcept;
  ::rust::Vec<::concrete_optimizer::dag::OperatorIndex> get_output_indices() const noexcept;
  ~Dag() = delete;

private:
  friend ::rust::layout;
  struct layout {
    static ::std::size_t size() noexcept;
    static ::std::size_t align() noexcept;
  };
};
#endif // CXXBRIDGE1_STRUCT_concrete_optimizer$Dag

#ifndef CXXBRIDGE1_STRUCT_concrete_optimizer$DagBuilder
#define CXXBRIDGE1_STRUCT_concrete_optimizer$DagBuilder
struct DagBuilder final : public ::rust::Opaque {
  ::rust::String dump() const noexcept;
  ::concrete_optimizer::dag::OperatorIndex add_input(::std::uint8_t out_precision, ::rust::Slice<::std::uint64_t const> out_shape, ::concrete_optimizer::Location const &location) noexcept;
  ::concrete_optimizer::dag::OperatorIndex add_zero_noise(::std::uint8_t out_precision, ::rust::Slice<::std::uint64_t const> out_shape, ::concrete_optimizer::Location const &location) noexcept;
  ::concrete_optimizer::dag::OperatorIndex add_lut(::concrete_optimizer::dag::OperatorIndex input, ::rust::Slice<::std::uint64_t const> table, ::std::uint8_t out_precision, ::concrete_optimizer::Location const &location) noexcept;
  ::concrete_optimizer::dag::OperatorIndex add_dot(::rust::Slice<::concrete_optimizer::dag::OperatorIndex const> inputs, ::rust::Box<::concrete_optimizer::Weights> weights, ::concrete_optimizer::Location const &location) noexcept;
  ::concrete_optimizer::dag::OperatorIndex add_linear_noise(::rust::Slice<::concrete_optimizer::dag::OperatorIndex const> inputs, double lwe_dim_cost_factor, double fixed_cost, ::rust::Slice<double const> weights, ::rust::Slice<::std::uint64_t const> out_shape, ::rust::Str comment, ::concrete_optimizer::Location const &location) noexcept;
  ::concrete_optimizer::dag::OperatorIndex add_max_noise(::rust::Slice<::concrete_optimizer::dag::OperatorIndex const> inputs, ::rust::Slice<::std::uint64_t const> out_shape, ::concrete_optimizer::Location const &location) noexcept;
  ::concrete_optimizer::dag::OperatorIndex add_round_op(::concrete_optimizer::dag::OperatorIndex input, ::std::uint8_t rounded_precision, ::concrete_optimizer::Location const &location) noexcept;
  ::concrete_optimizer::dag::OperatorIndex add_unsafe_cast_op(::concrete_optimizer::dag::OperatorIndex input, ::std::uint8_t rounded_precision, ::concrete_optimizer::Location const &location) noexcept;
  ::concrete_optimizer::dag::OperatorIndex add_change_partition_with_src(::concrete_optimizer::dag::OperatorIndex input, ::concrete_optimizer::ExternalPartition const &src_partition, ::concrete_optimizer::Location const &location) noexcept;
  ::concrete_optimizer::dag::OperatorIndex add_change_partition_with_dst(::concrete_optimizer::dag::OperatorIndex input, ::concrete_optimizer::ExternalPartition const &dst_partition, ::concrete_optimizer::Location const &location) noexcept;
  void tag_operator_as_output(::concrete_optimizer::dag::OperatorIndex op) noexcept;
  ~DagBuilder() = delete;

private:
  friend ::rust::layout;
  struct layout {
    static ::std::size_t size() noexcept;
    static ::std::size_t align() noexcept;
  };
};
#endif // CXXBRIDGE1_STRUCT_concrete_optimizer$DagBuilder

#ifndef CXXBRIDGE1_STRUCT_concrete_optimizer$Location
#define CXXBRIDGE1_STRUCT_concrete_optimizer$Location
struct Location final : public ::rust::Opaque {
  ~Location() = delete;

private:
  friend ::rust::layout;
  struct layout {
    static ::std::size_t size() noexcept;
    static ::std::size_t align() noexcept;
  };
};
#endif // CXXBRIDGE1_STRUCT_concrete_optimizer$Location

#ifndef CXXBRIDGE1_STRUCT_concrete_optimizer$ExternalPartition
#define CXXBRIDGE1_STRUCT_concrete_optimizer$ExternalPartition
struct ExternalPartition final : public ::rust::Opaque {
  ~ExternalPartition() = delete;

private:
  friend ::rust::layout;
  struct layout {
    static ::std::size_t size() noexcept;
    static ::std::size_t align() noexcept;
  };
};
#endif // CXXBRIDGE1_STRUCT_concrete_optimizer$ExternalPartition

#ifndef CXXBRIDGE1_STRUCT_concrete_optimizer$Weights
#define CXXBRIDGE1_STRUCT_concrete_optimizer$Weights
struct Weights final : public ::rust::Opaque {
  ~Weights() = delete;

private:
  friend ::rust::layout;
  struct layout {
    static ::std::size_t size() noexcept;
    static ::std::size_t align() noexcept;
  };
};
#endif // CXXBRIDGE1_STRUCT_concrete_optimizer$Weights

#ifndef CXXBRIDGE1_ENUM_concrete_optimizer$Encoding
#define CXXBRIDGE1_ENUM_concrete_optimizer$Encoding
enum class Encoding : ::std::uint8_t {
  Auto = 0,
  Native = 1,
  Crt = 2,
};
#endif // CXXBRIDGE1_ENUM_concrete_optimizer$Encoding

namespace dag {
#ifndef CXXBRIDGE1_STRUCT_concrete_optimizer$dag$OperatorIndex
#define CXXBRIDGE1_STRUCT_concrete_optimizer$dag$OperatorIndex
struct OperatorIndex final {
  ::std::size_t index;

  using IsRelocatable = ::std::true_type;
};
#endif // CXXBRIDGE1_STRUCT_concrete_optimizer$dag$OperatorIndex
} // namespace dag

namespace v0 {
#ifndef CXXBRIDGE1_STRUCT_concrete_optimizer$v0$Solution
#define CXXBRIDGE1_STRUCT_concrete_optimizer$v0$Solution
struct Solution final {
  ::std::uint64_t input_lwe_dimension;
  ::std::uint64_t internal_ks_output_lwe_dimension;
  ::std::uint64_t ks_decomposition_level_count;
  ::std::uint64_t ks_decomposition_base_log;
  ::std::uint64_t glwe_polynomial_size;
  ::std::uint64_t glwe_dimension;
  ::std::uint64_t br_decomposition_level_count;
  ::std::uint64_t br_decomposition_base_log;
  double complexity;
  double noise_max;
  double p_error;

  using IsRelocatable = ::std::true_type;
};
#endif // CXXBRIDGE1_STRUCT_concrete_optimizer$v0$Solution
} // namespace v0

namespace dag {
#ifndef CXXBRIDGE1_STRUCT_concrete_optimizer$dag$DagSolution
#define CXXBRIDGE1_STRUCT_concrete_optimizer$dag$DagSolution
struct DagSolution final {
  ::std::uint64_t input_lwe_dimension;
  ::std::uint64_t internal_ks_output_lwe_dimension;
  ::std::uint64_t ks_decomposition_level_count;
  ::std::uint64_t ks_decomposition_base_log;
  ::std::uint64_t glwe_polynomial_size;
  ::std::uint64_t glwe_dimension;
  ::std::uint64_t br_decomposition_level_count;
  ::std::uint64_t br_decomposition_base_log;
  double complexity;
  double noise_max;
  double p_error;
  double global_p_error;
  bool use_wop_pbs;
  ::std::uint64_t cb_decomposition_level_count;
  ::std::uint64_t cb_decomposition_base_log;
  ::std::uint64_t pp_decomposition_level_count;
  ::std::uint64_t pp_decomposition_base_log;
  ::rust::Vec<::std::uint64_t> crt_decomposition;

  using IsRelocatable = ::std::true_type;
};
#endif // CXXBRIDGE1_STRUCT_concrete_optimizer$dag$DagSolution
} // namespace dag

#ifndef CXXBRIDGE1_ENUM_concrete_optimizer$MultiParamStrategy
#define CXXBRIDGE1_ENUM_concrete_optimizer$MultiParamStrategy
enum class MultiParamStrategy : ::std::uint8_t {
  ByPrecision = 0,
  ByPrecisionAndNorm2 = 1,
};
#endif // CXXBRIDGE1_ENUM_concrete_optimizer$MultiParamStrategy

namespace restriction {
#ifndef CXXBRIDGE1_STRUCT_concrete_optimizer$restriction$RangeRestriction
#define CXXBRIDGE1_STRUCT_concrete_optimizer$restriction$RangeRestriction
struct RangeRestriction final {
  ::rust::Vec<::std::uint64_t> glwe_log_polynomial_sizes;
  ::rust::Vec<::std::uint64_t> glwe_dimensions;
  ::rust::Vec<::std::uint64_t> internal_lwe_dimensions;
  ::rust::Vec<::std::uint64_t> pbs_level_count;
  ::rust::Vec<::std::uint64_t> pbs_base_log;
  ::rust::Vec<::std::uint64_t> ks_level_count;
  ::rust::Vec<::std::uint64_t> ks_base_log;

  ::rust::String range_restriction_to_json() const noexcept;
  using IsRelocatable = ::std::true_type;
};
#endif // CXXBRIDGE1_STRUCT_concrete_optimizer$restriction$RangeRestriction
} // namespace restriction

#ifndef CXXBRIDGE1_STRUCT_concrete_optimizer$Options
#define CXXBRIDGE1_STRUCT_concrete_optimizer$Options
struct Options final {
  ::std::uint64_t security_level;
  double maximum_acceptable_error_probability;
  bool key_sharing;
  ::concrete_optimizer::MultiParamStrategy multi_param_strategy;
  double default_log_norm2_woppbs;
  bool use_gpu_constraints;
  ::concrete_optimizer::Encoding encoding;
  bool cache_on_disk;
  ::std::uint32_t ciphertext_modulus_log;
  ::std::uint32_t fft_precision;
  ::std::shared_ptr<::concrete_optimizer::restriction::RangeRestriction> range_restriction;
  ::std::shared_ptr<::concrete_optimizer::restriction::KeysetRestriction> keyset_restriction;

  using IsRelocatable = ::std::true_type;
};
#endif // CXXBRIDGE1_STRUCT_concrete_optimizer$Options

namespace dag {
#ifndef CXXBRIDGE1_STRUCT_concrete_optimizer$dag$BrDecompositionParameters
#define CXXBRIDGE1_STRUCT_concrete_optimizer$dag$BrDecompositionParameters
struct BrDecompositionParameters final {
  ::std::uint64_t level;
  ::std::uint64_t log2_base;

  using IsRelocatable = ::std::true_type;
};
#endif // CXXBRIDGE1_STRUCT_concrete_optimizer$dag$BrDecompositionParameters

#ifndef CXXBRIDGE1_STRUCT_concrete_optimizer$dag$KsDecompositionParameters
#define CXXBRIDGE1_STRUCT_concrete_optimizer$dag$KsDecompositionParameters
struct KsDecompositionParameters final {
  ::std::uint64_t level;
  ::std::uint64_t log2_base;

  using IsRelocatable = ::std::true_type;
};
#endif // CXXBRIDGE1_STRUCT_concrete_optimizer$dag$KsDecompositionParameters

#ifndef CXXBRIDGE1_STRUCT_concrete_optimizer$dag$SecretLweKey
#define CXXBRIDGE1_STRUCT_concrete_optimizer$dag$SecretLweKey
struct SecretLweKey final {
  ::std::uint64_t identifier;
  ::std::uint64_t polynomial_size;
  ::std::uint64_t glwe_dimension;
  ::rust::String description;

  using IsRelocatable = ::std::true_type;
};
#endif // CXXBRIDGE1_STRUCT_concrete_optimizer$dag$SecretLweKey

#ifndef CXXBRIDGE1_STRUCT_concrete_optimizer$dag$BootstrapKey
#define CXXBRIDGE1_STRUCT_concrete_optimizer$dag$BootstrapKey
struct BootstrapKey final {
  ::std::uint64_t identifier;
  ::concrete_optimizer::dag::SecretLweKey input_key;
  ::concrete_optimizer::dag::SecretLweKey output_key;
  ::concrete_optimizer::dag::BrDecompositionParameters br_decomposition_parameter;
  ::rust::String description;

  using IsRelocatable = ::std::true_type;
};
#endif // CXXBRIDGE1_STRUCT_concrete_optimizer$dag$BootstrapKey

#ifndef CXXBRIDGE1_STRUCT_concrete_optimizer$dag$KeySwitchKey
#define CXXBRIDGE1_STRUCT_concrete_optimizer$dag$KeySwitchKey
struct KeySwitchKey final {
  ::std::uint64_t identifier;
  ::concrete_optimizer::dag::SecretLweKey input_key;
  ::concrete_optimizer::dag::SecretLweKey output_key;
  ::concrete_optimizer::dag::KsDecompositionParameters ks_decomposition_parameter;
  ::rust::String description;

  using IsRelocatable = ::std::true_type;
};
#endif // CXXBRIDGE1_STRUCT_concrete_optimizer$dag$KeySwitchKey

#ifndef CXXBRIDGE1_STRUCT_concrete_optimizer$dag$ConversionKeySwitchKey
#define CXXBRIDGE1_STRUCT_concrete_optimizer$dag$ConversionKeySwitchKey
struct ConversionKeySwitchKey final {
  ::std::uint64_t identifier;
  ::concrete_optimizer::dag::SecretLweKey input_key;
  ::concrete_optimizer::dag::SecretLweKey output_key;
  ::concrete_optimizer::dag::KsDecompositionParameters ks_decomposition_parameter;
  bool fast_keyswitch;
  ::rust::String description;

  using IsRelocatable = ::std::true_type;
};
#endif // CXXBRIDGE1_STRUCT_concrete_optimizer$dag$ConversionKeySwitchKey

#ifndef CXXBRIDGE1_STRUCT_concrete_optimizer$dag$CircuitBoostrapKey
#define CXXBRIDGE1_STRUCT_concrete_optimizer$dag$CircuitBoostrapKey
struct CircuitBoostrapKey final {
  ::std::uint64_t identifier;
  ::concrete_optimizer::dag::SecretLweKey representation_key;
  ::concrete_optimizer::dag::BrDecompositionParameters br_decomposition_parameter;
  ::rust::String description;

  using IsRelocatable = ::std::true_type;
};
#endif // CXXBRIDGE1_STRUCT_concrete_optimizer$dag$CircuitBoostrapKey
} // namespace dag
} // namespace concrete_optimizer

#ifndef CXXBRIDGE1_STRUCT_PrivateFunctionalPackingBoostrapKey
#define CXXBRIDGE1_STRUCT_PrivateFunctionalPackingBoostrapKey
struct PrivateFunctionalPackingBoostrapKey final {
  ::std::uint64_t identifier;
  ::concrete_optimizer::dag::SecretLweKey representation_key;
  ::concrete_optimizer::dag::BrDecompositionParameters br_decomposition_parameter;
  ::rust::String description;

  using IsRelocatable = ::std::true_type;
};
#endif // CXXBRIDGE1_STRUCT_PrivateFunctionalPackingBoostrapKey

#ifndef CXXBRIDGE1_STRUCT_CircuitKeys
#define CXXBRIDGE1_STRUCT_CircuitKeys
struct CircuitKeys final {
  ::rust::Vec<::concrete_optimizer::dag::SecretLweKey> secret_keys;
  ::rust::Vec<::concrete_optimizer::dag::KeySwitchKey> keyswitch_keys;
  ::rust::Vec<::concrete_optimizer::dag::BootstrapKey> bootstrap_keys;
  ::rust::Vec<::concrete_optimizer::dag::ConversionKeySwitchKey> conversion_keyswitch_keys;
  ::rust::Vec<::concrete_optimizer::dag::CircuitBoostrapKey> circuit_bootstrap_keys;
  ::rust::Vec<::PrivateFunctionalPackingBoostrapKey> private_functional_packing_keys;

  using IsRelocatable = ::std::true_type;
};
#endif // CXXBRIDGE1_STRUCT_CircuitKeys

namespace concrete_optimizer {
namespace dag {
#ifndef CXXBRIDGE1_STRUCT_concrete_optimizer$dag$InstructionKeys
#define CXXBRIDGE1_STRUCT_concrete_optimizer$dag$InstructionKeys
struct InstructionKeys final {
  ::std::uint64_t input_key;
  ::std::uint64_t tlu_keyswitch_key;
  ::std::uint64_t tlu_bootstrap_key;
  ::std::uint64_t tlu_circuit_bootstrap_key;
  ::std::uint64_t tlu_private_functional_packing_key;
  ::std::uint64_t output_key;
  ::rust::Vec<::std::uint64_t> extra_conversion_keys;

  using IsRelocatable = ::std::true_type;
};
#endif // CXXBRIDGE1_STRUCT_concrete_optimizer$dag$InstructionKeys

#ifndef CXXBRIDGE1_STRUCT_concrete_optimizer$dag$CircuitSolution
#define CXXBRIDGE1_STRUCT_concrete_optimizer$dag$CircuitSolution
struct CircuitSolution final {
  ::CircuitKeys circuit_keys;
  ::rust::Vec<::concrete_optimizer::dag::InstructionKeys> instructions_keys;
  ::rust::Vec<::std::uint64_t> crt_decomposition;
  double complexity;
  double p_error;
  double global_p_error;
  bool is_feasible;
  ::rust::String error_msg;

  ::rust::String dump() const noexcept;
  ::rust::String short_dump() const noexcept;
  using IsRelocatable = ::std::true_type;
};
#endif // CXXBRIDGE1_STRUCT_concrete_optimizer$dag$CircuitSolution
} // namespace dag

namespace restriction {
#ifndef CXXBRIDGE1_STRUCT_concrete_optimizer$restriction$LweSecretKeyInfo
#define CXXBRIDGE1_STRUCT_concrete_optimizer$restriction$LweSecretKeyInfo
struct LweSecretKeyInfo final {
  ::std::uint64_t lwe_dimension;

  using IsRelocatable = ::std::true_type;
};
#endif // CXXBRIDGE1_STRUCT_concrete_optimizer$restriction$LweSecretKeyInfo

#ifndef CXXBRIDGE1_STRUCT_concrete_optimizer$restriction$LweBootstrapKeyInfo
#define CXXBRIDGE1_STRUCT_concrete_optimizer$restriction$LweBootstrapKeyInfo
struct LweBootstrapKeyInfo final {
  ::std::uint64_t level_count;
  ::std::uint64_t base_log;
  ::std::uint64_t glwe_dimension;
  ::std::uint64_t polynomial_size;
  ::std::uint64_t input_lwe_dimension;

  using IsRelocatable = ::std::true_type;
};
#endif // CXXBRIDGE1_STRUCT_concrete_optimizer$restriction$LweBootstrapKeyInfo

#ifndef CXXBRIDGE1_STRUCT_concrete_optimizer$restriction$LweKeyswitchKeyInfo
#define CXXBRIDGE1_STRUCT_concrete_optimizer$restriction$LweKeyswitchKeyInfo
struct LweKeyswitchKeyInfo final {
  ::std::uint64_t level_count;
  ::std::uint64_t base_log;
  ::std::uint64_t input_lwe_dimension;
  ::std::uint64_t output_lwe_dimension;

  using IsRelocatable = ::std::true_type;
};
#endif // CXXBRIDGE1_STRUCT_concrete_optimizer$restriction$LweKeyswitchKeyInfo

#ifndef CXXBRIDGE1_STRUCT_concrete_optimizer$restriction$KeysetInfo
#define CXXBRIDGE1_STRUCT_concrete_optimizer$restriction$KeysetInfo
struct KeysetInfo final {
  ::rust::Vec<::concrete_optimizer::restriction::LweSecretKeyInfo> lwe_secret_keys;
  ::rust::Vec<::concrete_optimizer::restriction::LweBootstrapKeyInfo> lwe_bootstrap_keys;
  ::rust::Vec<::concrete_optimizer::restriction::LweKeyswitchKeyInfo> lwe_keyswitch_keys;

  using IsRelocatable = ::std::true_type;
};
#endif // CXXBRIDGE1_STRUCT_concrete_optimizer$restriction$KeysetInfo

#ifndef CXXBRIDGE1_STRUCT_concrete_optimizer$restriction$KeysetRestriction
#define CXXBRIDGE1_STRUCT_concrete_optimizer$restriction$KeysetRestriction
struct KeysetRestriction final {
  ::concrete_optimizer::restriction::KeysetInfo info;

  ::rust::String keyset_restriction_to_json() const noexcept;
  using IsRelocatable = ::std::true_type;
};
#endif // CXXBRIDGE1_STRUCT_concrete_optimizer$restriction$KeysetRestriction
} // namespace restriction

namespace utils {
#ifndef CXXBRIDGE1_STRUCT_concrete_optimizer$utils$PartitionDefinition
#define CXXBRIDGE1_STRUCT_concrete_optimizer$utils$PartitionDefinition
struct PartitionDefinition final {
  ::std::uint8_t precision;
  double norm2;

  using IsRelocatable = ::std::true_type;
};
#endif // CXXBRIDGE1_STRUCT_concrete_optimizer$utils$PartitionDefinition
} // namespace utils

namespace v0 {
extern "C" {
::concrete_optimizer::v0::Solution concrete_optimizer$v0$cxxbridge1$optimize_bootstrap(::std::uint64_t precision, double noise_factor, ::concrete_optimizer::Options const &options) noexcept;
} // extern "C"
} // namespace v0

namespace utils {
extern "C" {
void concrete_optimizer$utils$cxxbridge1$convert_to_dag_solution(::concrete_optimizer::v0::Solution const &solution, ::concrete_optimizer::dag::DagSolution *return$) noexcept;

void concrete_optimizer$utils$cxxbridge1$convert_to_circuit_solution(::concrete_optimizer::dag::DagSolution const &solution, ::concrete_optimizer::Dag const &dag, ::concrete_optimizer::dag::CircuitSolution *return$) noexcept;
} // extern "C"
} // namespace utils

extern "C" {
::std::size_t concrete_optimizer$cxxbridge1$Dag$operator$sizeof() noexcept;
::std::size_t concrete_optimizer$cxxbridge1$Dag$operator$alignof() noexcept;
::std::size_t concrete_optimizer$cxxbridge1$DagBuilder$operator$sizeof() noexcept;
::std::size_t concrete_optimizer$cxxbridge1$DagBuilder$operator$alignof() noexcept;
::std::size_t concrete_optimizer$cxxbridge1$Location$operator$sizeof() noexcept;
::std::size_t concrete_optimizer$cxxbridge1$Location$operator$alignof() noexcept;
::std::size_t concrete_optimizer$cxxbridge1$ExternalPartition$operator$sizeof() noexcept;
::std::size_t concrete_optimizer$cxxbridge1$ExternalPartition$operator$alignof() noexcept;
} // extern "C"

namespace utils {
extern "C" {
::concrete_optimizer::Location *concrete_optimizer$utils$cxxbridge1$location_unknown() noexcept;

::concrete_optimizer::Location *concrete_optimizer$utils$cxxbridge1$location_from_string(::rust::Str string) noexcept;

void concrete_optimizer$utils$cxxbridge1$generate_virtual_keyset_info(::rust::Vec<::concrete_optimizer::utils::PartitionDefinition> *partitions, bool generate_fks, ::concrete_optimizer::Options const &options, ::CircuitKeys *return$) noexcept;

::concrete_optimizer::ExternalPartition *concrete_optimizer$utils$cxxbridge1$get_external_partition(::rust::String *name, ::std::uint64_t log2_polynomial_size, ::std::uint64_t glwe_dimension, ::std::uint64_t internal_dim, double max_variance, double variance) noexcept;

double concrete_optimizer$utils$cxxbridge1$get_noise_br(::concrete_optimizer::Options const &options, ::std::uint64_t log2_polynomial_size, ::std::uint64_t glwe_dimension, ::std::uint64_t lwe_dim, ::std::uint64_t pbs_level, ::std::uint64_t pbs_log2_base) noexcept;
} // extern "C"
} // namespace utils

namespace dag {
extern "C" {
::concrete_optimizer::Dag *concrete_optimizer$dag$cxxbridge1$empty() noexcept;
} // extern "C"
} // namespace dag

extern "C" {
::concrete_optimizer::DagBuilder *concrete_optimizer$cxxbridge1$Dag$builder(::concrete_optimizer::Dag &self, ::rust::String *circuit) noexcept;

void concrete_optimizer$cxxbridge1$Dag$dump(::concrete_optimizer::Dag const &self, ::rust::String *return$) noexcept;

void concrete_optimizer$cxxbridge1$DagBuilder$dump(::concrete_optimizer::DagBuilder const &self, ::rust::String *return$) noexcept;

::concrete_optimizer::dag::OperatorIndex concrete_optimizer$cxxbridge1$DagBuilder$add_input(::concrete_optimizer::DagBuilder &self, ::std::uint8_t out_precision, ::rust::Slice<::std::uint64_t const> out_shape, ::concrete_optimizer::Location const &location) noexcept;

::concrete_optimizer::dag::OperatorIndex concrete_optimizer$cxxbridge1$DagBuilder$add_zero_noise(::concrete_optimizer::DagBuilder &self, ::std::uint8_t out_precision, ::rust::Slice<::std::uint64_t const> out_shape, ::concrete_optimizer::Location const &location) noexcept;

::concrete_optimizer::dag::OperatorIndex concrete_optimizer$cxxbridge1$DagBuilder$add_lut(::concrete_optimizer::DagBuilder &self, ::concrete_optimizer::dag::OperatorIndex input, ::rust::Slice<::std::uint64_t const> table, ::std::uint8_t out_precision, ::concrete_optimizer::Location const &location) noexcept;

::concrete_optimizer::dag::OperatorIndex concrete_optimizer$cxxbridge1$DagBuilder$add_dot(::concrete_optimizer::DagBuilder &self, ::rust::Slice<::concrete_optimizer::dag::OperatorIndex const> inputs, ::concrete_optimizer::Weights *weights, ::concrete_optimizer::Location const &location) noexcept;

::concrete_optimizer::dag::OperatorIndex concrete_optimizer$cxxbridge1$DagBuilder$add_linear_noise(::concrete_optimizer::DagBuilder &self, ::rust::Slice<::concrete_optimizer::dag::OperatorIndex const> inputs, double lwe_dim_cost_factor, double fixed_cost, ::rust::Slice<double const> weights, ::rust::Slice<::std::uint64_t const> out_shape, ::rust::Str comment, ::concrete_optimizer::Location const &location) noexcept;

::concrete_optimizer::dag::OperatorIndex concrete_optimizer$cxxbridge1$DagBuilder$add_max_noise(::concrete_optimizer::DagBuilder &self, ::rust::Slice<::concrete_optimizer::dag::OperatorIndex const> inputs, ::rust::Slice<::std::uint64_t const> out_shape, ::concrete_optimizer::Location const &location) noexcept;

::concrete_optimizer::dag::OperatorIndex concrete_optimizer$cxxbridge1$DagBuilder$add_round_op(::concrete_optimizer::DagBuilder &self, ::concrete_optimizer::dag::OperatorIndex input, ::std::uint8_t rounded_precision, ::concrete_optimizer::Location const &location) noexcept;

::concrete_optimizer::dag::OperatorIndex concrete_optimizer$cxxbridge1$DagBuilder$add_unsafe_cast_op(::concrete_optimizer::DagBuilder &self, ::concrete_optimizer::dag::OperatorIndex input, ::std::uint8_t rounded_precision, ::concrete_optimizer::Location const &location) noexcept;

::concrete_optimizer::dag::OperatorIndex concrete_optimizer$cxxbridge1$DagBuilder$add_change_partition_with_src(::concrete_optimizer::DagBuilder &self, ::concrete_optimizer::dag::OperatorIndex input, ::concrete_optimizer::ExternalPartition const &src_partition, ::concrete_optimizer::Location const &location) noexcept;

::concrete_optimizer::dag::OperatorIndex concrete_optimizer$cxxbridge1$DagBuilder$add_change_partition_with_dst(::concrete_optimizer::DagBuilder &self, ::concrete_optimizer::dag::OperatorIndex input, ::concrete_optimizer::ExternalPartition const &dst_partition, ::concrete_optimizer::Location const &location) noexcept;

void concrete_optimizer$cxxbridge1$DagBuilder$tag_operator_as_output(::concrete_optimizer::DagBuilder &self, ::concrete_optimizer::dag::OperatorIndex op) noexcept;

void concrete_optimizer$cxxbridge1$Dag$optimize(::concrete_optimizer::Dag const &self, ::concrete_optimizer::Options const &options, ::concrete_optimizer::dag::DagSolution *return$) noexcept;

void concrete_optimizer$cxxbridge1$Dag$add_composition(::concrete_optimizer::Dag &self, ::std::string const &from_func, ::std::size_t from_pos, ::std::string const &to_func, ::std::size_t to_pos) noexcept;

void concrete_optimizer$cxxbridge1$Dag$add_all_compositions(::concrete_optimizer::Dag &self) noexcept;
} // extern "C"

namespace dag {
extern "C" {
void concrete_optimizer$dag$cxxbridge1$CircuitSolution$dump(::concrete_optimizer::dag::CircuitSolution const &self, ::rust::String *return$) noexcept;

void concrete_optimizer$dag$cxxbridge1$CircuitSolution$short_dump(::concrete_optimizer::dag::CircuitSolution const &self, ::rust::String *return$) noexcept;
} // extern "C"
} // namespace dag

extern "C" {
::std::size_t concrete_optimizer$cxxbridge1$Weights$operator$sizeof() noexcept;
::std::size_t concrete_optimizer$cxxbridge1$Weights$operator$alignof() noexcept;
} // extern "C"

namespace weights {
extern "C" {
::concrete_optimizer::Weights *concrete_optimizer$weights$cxxbridge1$vector(::rust::Slice<::std::int64_t const> weights) noexcept;

::concrete_optimizer::Weights *concrete_optimizer$weights$cxxbridge1$number(::std::int64_t weight) noexcept;
} // extern "C"
} // namespace weights

extern "C" {
::std::size_t concrete_optimizer$cxxbridge1$Dag$get_circuit_count(::concrete_optimizer::Dag const &self) noexcept;

void concrete_optimizer$cxxbridge1$Dag$optimize_multi(::concrete_optimizer::Dag const &self, ::concrete_optimizer::Options const &options, ::concrete_optimizer::dag::CircuitSolution *return$) noexcept;

void concrete_optimizer$cxxbridge1$Dag$get_input_indices(::concrete_optimizer::Dag const &self, ::rust::Vec<::concrete_optimizer::dag::OperatorIndex> *return$) noexcept;

void concrete_optimizer$cxxbridge1$Dag$get_output_indices(::concrete_optimizer::Dag const &self, ::rust::Vec<::concrete_optimizer::dag::OperatorIndex> *return$) noexcept;

::std::uint64_t concrete_optimizer$cxxbridge1$NO_KEY_ID() noexcept;
} // extern "C"

namespace restriction {
extern "C" {
void concrete_optimizer$restriction$cxxbridge1$RangeRestriction$range_restriction_to_json(::concrete_optimizer::restriction::RangeRestriction const &self, ::rust::String *return$) noexcept;

void concrete_optimizer$restriction$cxxbridge1$range_restriction_from_json(::rust::Str input, ::concrete_optimizer::restriction::RangeRestriction *return$) noexcept;

void concrete_optimizer$restriction$cxxbridge1$KeysetRestriction$keyset_restriction_to_json(::concrete_optimizer::restriction::KeysetRestriction const &self, ::rust::String *return$) noexcept;

void concrete_optimizer$restriction$cxxbridge1$keyset_restriction_from_json(::rust::Str input, ::concrete_optimizer::restriction::KeysetRestriction *return$) noexcept;
} // extern "C"
} // namespace restriction

namespace v0 {
::concrete_optimizer::v0::Solution optimize_bootstrap(::std::uint64_t precision, double noise_factor, ::concrete_optimizer::Options const &options) noexcept {
  return concrete_optimizer$v0$cxxbridge1$optimize_bootstrap(precision, noise_factor, options);
}
} // namespace v0

namespace utils {
::concrete_optimizer::dag::DagSolution convert_to_dag_solution(::concrete_optimizer::v0::Solution const &solution) noexcept {
  ::rust::MaybeUninit<::concrete_optimizer::dag::DagSolution> return$;
  concrete_optimizer$utils$cxxbridge1$convert_to_dag_solution(solution, &return$.value);
  return ::std::move(return$.value);
}

::concrete_optimizer::dag::CircuitSolution convert_to_circuit_solution(::concrete_optimizer::dag::DagSolution const &solution, ::concrete_optimizer::Dag const &dag) noexcept {
  ::rust::MaybeUninit<::concrete_optimizer::dag::CircuitSolution> return$;
  concrete_optimizer$utils$cxxbridge1$convert_to_circuit_solution(solution, dag, &return$.value);
  return ::std::move(return$.value);
}
} // namespace utils

::std::size_t Dag::layout::size() noexcept {
  return concrete_optimizer$cxxbridge1$Dag$operator$sizeof();
}

::std::size_t Dag::layout::align() noexcept {
  return concrete_optimizer$cxxbridge1$Dag$operator$alignof();
}

::std::size_t DagBuilder::layout::size() noexcept {
  return concrete_optimizer$cxxbridge1$DagBuilder$operator$sizeof();
}

::std::size_t DagBuilder::layout::align() noexcept {
  return concrete_optimizer$cxxbridge1$DagBuilder$operator$alignof();
}

::std::size_t Location::layout::size() noexcept {
  return concrete_optimizer$cxxbridge1$Location$operator$sizeof();
}

::std::size_t Location::layout::align() noexcept {
  return concrete_optimizer$cxxbridge1$Location$operator$alignof();
}

::std::size_t ExternalPartition::layout::size() noexcept {
  return concrete_optimizer$cxxbridge1$ExternalPartition$operator$sizeof();
}

::std::size_t ExternalPartition::layout::align() noexcept {
  return concrete_optimizer$cxxbridge1$ExternalPartition$operator$alignof();
}

namespace utils {
::rust::Box<::concrete_optimizer::Location> location_unknown() noexcept {
  return ::rust::Box<::concrete_optimizer::Location>::from_raw(concrete_optimizer$utils$cxxbridge1$location_unknown());
}

::rust::Box<::concrete_optimizer::Location> location_from_string(::rust::Str string) noexcept {
  return ::rust::Box<::concrete_optimizer::Location>::from_raw(concrete_optimizer$utils$cxxbridge1$location_from_string(string));
}

::CircuitKeys generate_virtual_keyset_info(::rust::Vec<::concrete_optimizer::utils::PartitionDefinition> partitions, bool generate_fks, ::concrete_optimizer::Options const &options) noexcept {
  ::rust::ManuallyDrop<::rust::Vec<::concrete_optimizer::utils::PartitionDefinition>> partitions$(::std::move(partitions));
  ::rust::MaybeUninit<::CircuitKeys> return$;
  concrete_optimizer$utils$cxxbridge1$generate_virtual_keyset_info(&partitions$.value, generate_fks, options, &return$.value);
  return ::std::move(return$.value);
}

::rust::Box<::concrete_optimizer::ExternalPartition> get_external_partition(::rust::String name, ::std::uint64_t log2_polynomial_size, ::std::uint64_t glwe_dimension, ::std::uint64_t internal_dim, double max_variance, double variance) noexcept {
  return ::rust::Box<::concrete_optimizer::ExternalPartition>::from_raw(concrete_optimizer$utils$cxxbridge1$get_external_partition(&name, log2_polynomial_size, glwe_dimension, internal_dim, max_variance, variance));
}

double get_noise_br(::concrete_optimizer::Options const &options, ::std::uint64_t log2_polynomial_size, ::std::uint64_t glwe_dimension, ::std::uint64_t lwe_dim, ::std::uint64_t pbs_level, ::std::uint64_t pbs_log2_base) noexcept {
  return concrete_optimizer$utils$cxxbridge1$get_noise_br(options, log2_polynomial_size, glwe_dimension, lwe_dim, pbs_level, pbs_log2_base);
}
} // namespace utils

namespace dag {
::rust::Box<::concrete_optimizer::Dag> empty() noexcept {
  return ::rust::Box<::concrete_optimizer::Dag>::from_raw(concrete_optimizer$dag$cxxbridge1$empty());
}
} // namespace dag

::rust::Box<::concrete_optimizer::DagBuilder> Dag::builder(::rust::String circuit) noexcept {
  return ::rust::Box<::concrete_optimizer::DagBuilder>::from_raw(concrete_optimizer$cxxbridge1$Dag$builder(*this, &circuit));
}

::rust::String Dag::dump() const noexcept {
  ::rust::MaybeUninit<::rust::String> return$;
  concrete_optimizer$cxxbridge1$Dag$dump(*this, &return$.value);
  return ::std::move(return$.value);
}

::rust::String DagBuilder::dump() const noexcept {
  ::rust::MaybeUninit<::rust::String> return$;
  concrete_optimizer$cxxbridge1$DagBuilder$dump(*this, &return$.value);
  return ::std::move(return$.value);
}

::concrete_optimizer::dag::OperatorIndex DagBuilder::add_input(::std::uint8_t out_precision, ::rust::Slice<::std::uint64_t const> out_shape, ::concrete_optimizer::Location const &location) noexcept {
  return concrete_optimizer$cxxbridge1$DagBuilder$add_input(*this, out_precision, out_shape, location);
}

::concrete_optimizer::dag::OperatorIndex DagBuilder::add_zero_noise(::std::uint8_t out_precision, ::rust::Slice<::std::uint64_t const> out_shape, ::concrete_optimizer::Location const &location) noexcept {
  return concrete_optimizer$cxxbridge1$DagBuilder$add_zero_noise(*this, out_precision, out_shape, location);
}

::concrete_optimizer::dag::OperatorIndex DagBuilder::add_lut(::concrete_optimizer::dag::OperatorIndex input, ::rust::Slice<::std::uint64_t const> table, ::std::uint8_t out_precision, ::concrete_optimizer::Location const &location) noexcept {
  return concrete_optimizer$cxxbridge1$DagBuilder$add_lut(*this, input, table, out_precision, location);
}

::concrete_optimizer::dag::OperatorIndex DagBuilder::add_dot(::rust::Slice<::concrete_optimizer::dag::OperatorIndex const> inputs, ::rust::Box<::concrete_optimizer::Weights> weights, ::concrete_optimizer::Location const &location) noexcept {
  return concrete_optimizer$cxxbridge1$DagBuilder$add_dot(*this, inputs, weights.into_raw(), location);
}

::concrete_optimizer::dag::OperatorIndex DagBuilder::add_linear_noise(::rust::Slice<::concrete_optimizer::dag::OperatorIndex const> inputs, double lwe_dim_cost_factor, double fixed_cost, ::rust::Slice<double const> weights, ::rust::Slice<::std::uint64_t const> out_shape, ::rust::Str comment, ::concrete_optimizer::Location const &location) noexcept {
  return concrete_optimizer$cxxbridge1$DagBuilder$add_linear_noise(*this, inputs, lwe_dim_cost_factor, fixed_cost, weights, out_shape, comment, location);
}

::concrete_optimizer::dag::OperatorIndex DagBuilder::add_max_noise(::rust::Slice<::concrete_optimizer::dag::OperatorIndex const> inputs, ::rust::Slice<::std::uint64_t const> out_shape, ::concrete_optimizer::Location const &location) noexcept {
  return concrete_optimizer$cxxbridge1$DagBuilder$add_max_noise(*this, inputs, out_shape, location);
}

::concrete_optimizer::dag::OperatorIndex DagBuilder::add_round_op(::concrete_optimizer::dag::OperatorIndex input, ::std::uint8_t rounded_precision, ::concrete_optimizer::Location const &location) noexcept {
  return concrete_optimizer$cxxbridge1$DagBuilder$add_round_op(*this, input, rounded_precision, location);
}

::concrete_optimizer::dag::OperatorIndex DagBuilder::add_unsafe_cast_op(::concrete_optimizer::dag::OperatorIndex input, ::std::uint8_t rounded_precision, ::concrete_optimizer::Location const &location) noexcept {
  return concrete_optimizer$cxxbridge1$DagBuilder$add_unsafe_cast_op(*this, input, rounded_precision, location);
}

::concrete_optimizer::dag::OperatorIndex DagBuilder::add_change_partition_with_src(::concrete_optimizer::dag::OperatorIndex input, ::concrete_optimizer::ExternalPartition const &src_partition, ::concrete_optimizer::Location const &location) noexcept {
  return concrete_optimizer$cxxbridge1$DagBuilder$add_change_partition_with_src(*this, input, src_partition, location);
}

::concrete_optimizer::dag::OperatorIndex DagBuilder::add_change_partition_with_dst(::concrete_optimizer::dag::OperatorIndex input, ::concrete_optimizer::ExternalPartition const &dst_partition, ::concrete_optimizer::Location const &location) noexcept {
  return concrete_optimizer$cxxbridge1$DagBuilder$add_change_partition_with_dst(*this, input, dst_partition, location);
}

void DagBuilder::tag_operator_as_output(::concrete_optimizer::dag::OperatorIndex op) noexcept {
  concrete_optimizer$cxxbridge1$DagBuilder$tag_operator_as_output(*this, op);
}

::concrete_optimizer::dag::DagSolution Dag::optimize(::concrete_optimizer::Options const &options) const noexcept {
  ::rust::MaybeUninit<::concrete_optimizer::dag::DagSolution> return$;
  concrete_optimizer$cxxbridge1$Dag$optimize(*this, options, &return$.value);
  return ::std::move(return$.value);
}

void Dag::add_composition(::std::string const &from_func, ::std::size_t from_pos, ::std::string const &to_func, ::std::size_t to_pos) noexcept {
  concrete_optimizer$cxxbridge1$Dag$add_composition(*this, from_func, from_pos, to_func, to_pos);
}

void Dag::add_all_compositions() noexcept {
  concrete_optimizer$cxxbridge1$Dag$add_all_compositions(*this);
}

namespace dag {
::rust::String CircuitSolution::dump() const noexcept {
  ::rust::MaybeUninit<::rust::String> return$;
  concrete_optimizer$dag$cxxbridge1$CircuitSolution$dump(*this, &return$.value);
  return ::std::move(return$.value);
}

::rust::String CircuitSolution::short_dump() const noexcept {
  ::rust::MaybeUninit<::rust::String> return$;
  concrete_optimizer$dag$cxxbridge1$CircuitSolution$short_dump(*this, &return$.value);
  return ::std::move(return$.value);
}
} // namespace dag

::std::size_t Weights::layout::size() noexcept {
  return concrete_optimizer$cxxbridge1$Weights$operator$sizeof();
}

::std::size_t Weights::layout::align() noexcept {
  return concrete_optimizer$cxxbridge1$Weights$operator$alignof();
}

namespace weights {
::rust::Box<::concrete_optimizer::Weights> vector(::rust::Slice<::std::int64_t const> weights) noexcept {
  return ::rust::Box<::concrete_optimizer::Weights>::from_raw(concrete_optimizer$weights$cxxbridge1$vector(weights));
}

::rust::Box<::concrete_optimizer::Weights> number(::std::int64_t weight) noexcept {
  return ::rust::Box<::concrete_optimizer::Weights>::from_raw(concrete_optimizer$weights$cxxbridge1$number(weight));
}
} // namespace weights

::std::size_t Dag::get_circuit_count() const noexcept {
  return concrete_optimizer$cxxbridge1$Dag$get_circuit_count(*this);
}

::concrete_optimizer::dag::CircuitSolution Dag::optimize_multi(::concrete_optimizer::Options const &options) const noexcept {
  ::rust::MaybeUninit<::concrete_optimizer::dag::CircuitSolution> return$;
  concrete_optimizer$cxxbridge1$Dag$optimize_multi(*this, options, &return$.value);
  return ::std::move(return$.value);
}

::rust::Vec<::concrete_optimizer::dag::OperatorIndex> Dag::get_input_indices() const noexcept {
  ::rust::MaybeUninit<::rust::Vec<::concrete_optimizer::dag::OperatorIndex>> return$;
  concrete_optimizer$cxxbridge1$Dag$get_input_indices(*this, &return$.value);
  return ::std::move(return$.value);
}

::rust::Vec<::concrete_optimizer::dag::OperatorIndex> Dag::get_output_indices() const noexcept {
  ::rust::MaybeUninit<::rust::Vec<::concrete_optimizer::dag::OperatorIndex>> return$;
  concrete_optimizer$cxxbridge1$Dag$get_output_indices(*this, &return$.value);
  return ::std::move(return$.value);
}

::std::uint64_t NO_KEY_ID() noexcept {
  return concrete_optimizer$cxxbridge1$NO_KEY_ID();
}

namespace restriction {
::rust::String RangeRestriction::range_restriction_to_json() const noexcept {
  ::rust::MaybeUninit<::rust::String> return$;
  concrete_optimizer$restriction$cxxbridge1$RangeRestriction$range_restriction_to_json(*this, &return$.value);
  return ::std::move(return$.value);
}

::concrete_optimizer::restriction::RangeRestriction range_restriction_from_json(::rust::Str input) noexcept {
  ::rust::MaybeUninit<::concrete_optimizer::restriction::RangeRestriction> return$;
  concrete_optimizer$restriction$cxxbridge1$range_restriction_from_json(input, &return$.value);
  return ::std::move(return$.value);
}

::rust::String KeysetRestriction::keyset_restriction_to_json() const noexcept {
  ::rust::MaybeUninit<::rust::String> return$;
  concrete_optimizer$restriction$cxxbridge1$KeysetRestriction$keyset_restriction_to_json(*this, &return$.value);
  return ::std::move(return$.value);
}

::concrete_optimizer::restriction::KeysetRestriction keyset_restriction_from_json(::rust::Str input) noexcept {
  ::rust::MaybeUninit<::concrete_optimizer::restriction::KeysetRestriction> return$;
  concrete_optimizer$restriction$cxxbridge1$keyset_restriction_from_json(input, &return$.value);
  return ::std::move(return$.value);
}
} // namespace restriction
} // namespace concrete_optimizer

extern "C" {
::concrete_optimizer::Location *cxxbridge1$box$concrete_optimizer$Location$alloc() noexcept;
void cxxbridge1$box$concrete_optimizer$Location$dealloc(::concrete_optimizer::Location *) noexcept;
void cxxbridge1$box$concrete_optimizer$Location$drop(::rust::Box<::concrete_optimizer::Location> *ptr) noexcept;

void cxxbridge1$rust_vec$concrete_optimizer$utils$PartitionDefinition$new(::rust::Vec<::concrete_optimizer::utils::PartitionDefinition> const *ptr) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$utils$PartitionDefinition$drop(::rust::Vec<::concrete_optimizer::utils::PartitionDefinition> *ptr) noexcept;
::std::size_t cxxbridge1$rust_vec$concrete_optimizer$utils$PartitionDefinition$len(::rust::Vec<::concrete_optimizer::utils::PartitionDefinition> const *ptr) noexcept;
::std::size_t cxxbridge1$rust_vec$concrete_optimizer$utils$PartitionDefinition$capacity(::rust::Vec<::concrete_optimizer::utils::PartitionDefinition> const *ptr) noexcept;
::concrete_optimizer::utils::PartitionDefinition const *cxxbridge1$rust_vec$concrete_optimizer$utils$PartitionDefinition$data(::rust::Vec<::concrete_optimizer::utils::PartitionDefinition> const *ptr) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$utils$PartitionDefinition$reserve_total(::rust::Vec<::concrete_optimizer::utils::PartitionDefinition> *ptr, ::std::size_t new_cap) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$utils$PartitionDefinition$set_len(::rust::Vec<::concrete_optimizer::utils::PartitionDefinition> *ptr, ::std::size_t len) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$utils$PartitionDefinition$truncate(::rust::Vec<::concrete_optimizer::utils::PartitionDefinition> *ptr, ::std::size_t len) noexcept;

::concrete_optimizer::ExternalPartition *cxxbridge1$box$concrete_optimizer$ExternalPartition$alloc() noexcept;
void cxxbridge1$box$concrete_optimizer$ExternalPartition$dealloc(::concrete_optimizer::ExternalPartition *) noexcept;
void cxxbridge1$box$concrete_optimizer$ExternalPartition$drop(::rust::Box<::concrete_optimizer::ExternalPartition> *ptr) noexcept;

::concrete_optimizer::Dag *cxxbridge1$box$concrete_optimizer$Dag$alloc() noexcept;
void cxxbridge1$box$concrete_optimizer$Dag$dealloc(::concrete_optimizer::Dag *) noexcept;
void cxxbridge1$box$concrete_optimizer$Dag$drop(::rust::Box<::concrete_optimizer::Dag> *ptr) noexcept;

::concrete_optimizer::DagBuilder *cxxbridge1$box$concrete_optimizer$DagBuilder$alloc() noexcept;
void cxxbridge1$box$concrete_optimizer$DagBuilder$dealloc(::concrete_optimizer::DagBuilder *) noexcept;
void cxxbridge1$box$concrete_optimizer$DagBuilder$drop(::rust::Box<::concrete_optimizer::DagBuilder> *ptr) noexcept;

::concrete_optimizer::Weights *cxxbridge1$box$concrete_optimizer$Weights$alloc() noexcept;
void cxxbridge1$box$concrete_optimizer$Weights$dealloc(::concrete_optimizer::Weights *) noexcept;
void cxxbridge1$box$concrete_optimizer$Weights$drop(::rust::Box<::concrete_optimizer::Weights> *ptr) noexcept;

void cxxbridge1$rust_vec$concrete_optimizer$dag$OperatorIndex$new(::rust::Vec<::concrete_optimizer::dag::OperatorIndex> const *ptr) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$OperatorIndex$drop(::rust::Vec<::concrete_optimizer::dag::OperatorIndex> *ptr) noexcept;
::std::size_t cxxbridge1$rust_vec$concrete_optimizer$dag$OperatorIndex$len(::rust::Vec<::concrete_optimizer::dag::OperatorIndex> const *ptr) noexcept;
::std::size_t cxxbridge1$rust_vec$concrete_optimizer$dag$OperatorIndex$capacity(::rust::Vec<::concrete_optimizer::dag::OperatorIndex> const *ptr) noexcept;
::concrete_optimizer::dag::OperatorIndex const *cxxbridge1$rust_vec$concrete_optimizer$dag$OperatorIndex$data(::rust::Vec<::concrete_optimizer::dag::OperatorIndex> const *ptr) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$OperatorIndex$reserve_total(::rust::Vec<::concrete_optimizer::dag::OperatorIndex> *ptr, ::std::size_t new_cap) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$OperatorIndex$set_len(::rust::Vec<::concrete_optimizer::dag::OperatorIndex> *ptr, ::std::size_t len) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$OperatorIndex$truncate(::rust::Vec<::concrete_optimizer::dag::OperatorIndex> *ptr, ::std::size_t len) noexcept;

static_assert(sizeof(::std::shared_ptr<::concrete_optimizer::restriction::RangeRestriction>) == 2 * sizeof(void *), "");
static_assert(alignof(::std::shared_ptr<::concrete_optimizer::restriction::RangeRestriction>) == alignof(void *), "");
void cxxbridge1$shared_ptr$concrete_optimizer$restriction$RangeRestriction$null(::std::shared_ptr<::concrete_optimizer::restriction::RangeRestriction> *ptr) noexcept {
  ::new (ptr) ::std::shared_ptr<::concrete_optimizer::restriction::RangeRestriction>();
}
::concrete_optimizer::restriction::RangeRestriction *cxxbridge1$shared_ptr$concrete_optimizer$restriction$RangeRestriction$uninit(::std::shared_ptr<::concrete_optimizer::restriction::RangeRestriction> *ptr) noexcept {
  ::concrete_optimizer::restriction::RangeRestriction *uninit = reinterpret_cast<::concrete_optimizer::restriction::RangeRestriction *>(new ::rust::MaybeUninit<::concrete_optimizer::restriction::RangeRestriction>);
  ::new (ptr) ::std::shared_ptr<::concrete_optimizer::restriction::RangeRestriction>(uninit);
  return uninit;
}
void cxxbridge1$shared_ptr$concrete_optimizer$restriction$RangeRestriction$clone(::std::shared_ptr<::concrete_optimizer::restriction::RangeRestriction> const &self, ::std::shared_ptr<::concrete_optimizer::restriction::RangeRestriction> *ptr) noexcept {
  ::new (ptr) ::std::shared_ptr<::concrete_optimizer::restriction::RangeRestriction>(self);
}
::concrete_optimizer::restriction::RangeRestriction const *cxxbridge1$shared_ptr$concrete_optimizer$restriction$RangeRestriction$get(::std::shared_ptr<::concrete_optimizer::restriction::RangeRestriction> const &self) noexcept {
  return self.get();
}
void cxxbridge1$shared_ptr$concrete_optimizer$restriction$RangeRestriction$drop(::std::shared_ptr<::concrete_optimizer::restriction::RangeRestriction> *self) noexcept {
  self->~shared_ptr();
}

static_assert(sizeof(::std::shared_ptr<::concrete_optimizer::restriction::KeysetRestriction>) == 2 * sizeof(void *), "");
static_assert(alignof(::std::shared_ptr<::concrete_optimizer::restriction::KeysetRestriction>) == alignof(void *), "");
void cxxbridge1$shared_ptr$concrete_optimizer$restriction$KeysetRestriction$null(::std::shared_ptr<::concrete_optimizer::restriction::KeysetRestriction> *ptr) noexcept {
  ::new (ptr) ::std::shared_ptr<::concrete_optimizer::restriction::KeysetRestriction>();
}
::concrete_optimizer::restriction::KeysetRestriction *cxxbridge1$shared_ptr$concrete_optimizer$restriction$KeysetRestriction$uninit(::std::shared_ptr<::concrete_optimizer::restriction::KeysetRestriction> *ptr) noexcept {
  ::concrete_optimizer::restriction::KeysetRestriction *uninit = reinterpret_cast<::concrete_optimizer::restriction::KeysetRestriction *>(new ::rust::MaybeUninit<::concrete_optimizer::restriction::KeysetRestriction>);
  ::new (ptr) ::std::shared_ptr<::concrete_optimizer::restriction::KeysetRestriction>(uninit);
  return uninit;
}
void cxxbridge1$shared_ptr$concrete_optimizer$restriction$KeysetRestriction$clone(::std::shared_ptr<::concrete_optimizer::restriction::KeysetRestriction> const &self, ::std::shared_ptr<::concrete_optimizer::restriction::KeysetRestriction> *ptr) noexcept {
  ::new (ptr) ::std::shared_ptr<::concrete_optimizer::restriction::KeysetRestriction>(self);
}
::concrete_optimizer::restriction::KeysetRestriction const *cxxbridge1$shared_ptr$concrete_optimizer$restriction$KeysetRestriction$get(::std::shared_ptr<::concrete_optimizer::restriction::KeysetRestriction> const &self) noexcept {
  return self.get();
}
void cxxbridge1$shared_ptr$concrete_optimizer$restriction$KeysetRestriction$drop(::std::shared_ptr<::concrete_optimizer::restriction::KeysetRestriction> *self) noexcept {
  self->~shared_ptr();
}

void cxxbridge1$rust_vec$concrete_optimizer$dag$SecretLweKey$new(::rust::Vec<::concrete_optimizer::dag::SecretLweKey> const *ptr) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$SecretLweKey$drop(::rust::Vec<::concrete_optimizer::dag::SecretLweKey> *ptr) noexcept;
::std::size_t cxxbridge1$rust_vec$concrete_optimizer$dag$SecretLweKey$len(::rust::Vec<::concrete_optimizer::dag::SecretLweKey> const *ptr) noexcept;
::std::size_t cxxbridge1$rust_vec$concrete_optimizer$dag$SecretLweKey$capacity(::rust::Vec<::concrete_optimizer::dag::SecretLweKey> const *ptr) noexcept;
::concrete_optimizer::dag::SecretLweKey const *cxxbridge1$rust_vec$concrete_optimizer$dag$SecretLweKey$data(::rust::Vec<::concrete_optimizer::dag::SecretLweKey> const *ptr) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$SecretLweKey$reserve_total(::rust::Vec<::concrete_optimizer::dag::SecretLweKey> *ptr, ::std::size_t new_cap) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$SecretLweKey$set_len(::rust::Vec<::concrete_optimizer::dag::SecretLweKey> *ptr, ::std::size_t len) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$SecretLweKey$truncate(::rust::Vec<::concrete_optimizer::dag::SecretLweKey> *ptr, ::std::size_t len) noexcept;

void cxxbridge1$rust_vec$concrete_optimizer$dag$KeySwitchKey$new(::rust::Vec<::concrete_optimizer::dag::KeySwitchKey> const *ptr) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$KeySwitchKey$drop(::rust::Vec<::concrete_optimizer::dag::KeySwitchKey> *ptr) noexcept;
::std::size_t cxxbridge1$rust_vec$concrete_optimizer$dag$KeySwitchKey$len(::rust::Vec<::concrete_optimizer::dag::KeySwitchKey> const *ptr) noexcept;
::std::size_t cxxbridge1$rust_vec$concrete_optimizer$dag$KeySwitchKey$capacity(::rust::Vec<::concrete_optimizer::dag::KeySwitchKey> const *ptr) noexcept;
::concrete_optimizer::dag::KeySwitchKey const *cxxbridge1$rust_vec$concrete_optimizer$dag$KeySwitchKey$data(::rust::Vec<::concrete_optimizer::dag::KeySwitchKey> const *ptr) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$KeySwitchKey$reserve_total(::rust::Vec<::concrete_optimizer::dag::KeySwitchKey> *ptr, ::std::size_t new_cap) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$KeySwitchKey$set_len(::rust::Vec<::concrete_optimizer::dag::KeySwitchKey> *ptr, ::std::size_t len) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$KeySwitchKey$truncate(::rust::Vec<::concrete_optimizer::dag::KeySwitchKey> *ptr, ::std::size_t len) noexcept;

void cxxbridge1$rust_vec$concrete_optimizer$dag$BootstrapKey$new(::rust::Vec<::concrete_optimizer::dag::BootstrapKey> const *ptr) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$BootstrapKey$drop(::rust::Vec<::concrete_optimizer::dag::BootstrapKey> *ptr) noexcept;
::std::size_t cxxbridge1$rust_vec$concrete_optimizer$dag$BootstrapKey$len(::rust::Vec<::concrete_optimizer::dag::BootstrapKey> const *ptr) noexcept;
::std::size_t cxxbridge1$rust_vec$concrete_optimizer$dag$BootstrapKey$capacity(::rust::Vec<::concrete_optimizer::dag::BootstrapKey> const *ptr) noexcept;
::concrete_optimizer::dag::BootstrapKey const *cxxbridge1$rust_vec$concrete_optimizer$dag$BootstrapKey$data(::rust::Vec<::concrete_optimizer::dag::BootstrapKey> const *ptr) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$BootstrapKey$reserve_total(::rust::Vec<::concrete_optimizer::dag::BootstrapKey> *ptr, ::std::size_t new_cap) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$BootstrapKey$set_len(::rust::Vec<::concrete_optimizer::dag::BootstrapKey> *ptr, ::std::size_t len) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$BootstrapKey$truncate(::rust::Vec<::concrete_optimizer::dag::BootstrapKey> *ptr, ::std::size_t len) noexcept;

void cxxbridge1$rust_vec$concrete_optimizer$dag$ConversionKeySwitchKey$new(::rust::Vec<::concrete_optimizer::dag::ConversionKeySwitchKey> const *ptr) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$ConversionKeySwitchKey$drop(::rust::Vec<::concrete_optimizer::dag::ConversionKeySwitchKey> *ptr) noexcept;
::std::size_t cxxbridge1$rust_vec$concrete_optimizer$dag$ConversionKeySwitchKey$len(::rust::Vec<::concrete_optimizer::dag::ConversionKeySwitchKey> const *ptr) noexcept;
::std::size_t cxxbridge1$rust_vec$concrete_optimizer$dag$ConversionKeySwitchKey$capacity(::rust::Vec<::concrete_optimizer::dag::ConversionKeySwitchKey> const *ptr) noexcept;
::concrete_optimizer::dag::ConversionKeySwitchKey const *cxxbridge1$rust_vec$concrete_optimizer$dag$ConversionKeySwitchKey$data(::rust::Vec<::concrete_optimizer::dag::ConversionKeySwitchKey> const *ptr) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$ConversionKeySwitchKey$reserve_total(::rust::Vec<::concrete_optimizer::dag::ConversionKeySwitchKey> *ptr, ::std::size_t new_cap) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$ConversionKeySwitchKey$set_len(::rust::Vec<::concrete_optimizer::dag::ConversionKeySwitchKey> *ptr, ::std::size_t len) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$ConversionKeySwitchKey$truncate(::rust::Vec<::concrete_optimizer::dag::ConversionKeySwitchKey> *ptr, ::std::size_t len) noexcept;

void cxxbridge1$rust_vec$concrete_optimizer$dag$CircuitBoostrapKey$new(::rust::Vec<::concrete_optimizer::dag::CircuitBoostrapKey> const *ptr) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$CircuitBoostrapKey$drop(::rust::Vec<::concrete_optimizer::dag::CircuitBoostrapKey> *ptr) noexcept;
::std::size_t cxxbridge1$rust_vec$concrete_optimizer$dag$CircuitBoostrapKey$len(::rust::Vec<::concrete_optimizer::dag::CircuitBoostrapKey> const *ptr) noexcept;
::std::size_t cxxbridge1$rust_vec$concrete_optimizer$dag$CircuitBoostrapKey$capacity(::rust::Vec<::concrete_optimizer::dag::CircuitBoostrapKey> const *ptr) noexcept;
::concrete_optimizer::dag::CircuitBoostrapKey const *cxxbridge1$rust_vec$concrete_optimizer$dag$CircuitBoostrapKey$data(::rust::Vec<::concrete_optimizer::dag::CircuitBoostrapKey> const *ptr) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$CircuitBoostrapKey$reserve_total(::rust::Vec<::concrete_optimizer::dag::CircuitBoostrapKey> *ptr, ::std::size_t new_cap) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$CircuitBoostrapKey$set_len(::rust::Vec<::concrete_optimizer::dag::CircuitBoostrapKey> *ptr, ::std::size_t len) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$CircuitBoostrapKey$truncate(::rust::Vec<::concrete_optimizer::dag::CircuitBoostrapKey> *ptr, ::std::size_t len) noexcept;

void cxxbridge1$rust_vec$PrivateFunctionalPackingBoostrapKey$new(::rust::Vec<::PrivateFunctionalPackingBoostrapKey> const *ptr) noexcept;
void cxxbridge1$rust_vec$PrivateFunctionalPackingBoostrapKey$drop(::rust::Vec<::PrivateFunctionalPackingBoostrapKey> *ptr) noexcept;
::std::size_t cxxbridge1$rust_vec$PrivateFunctionalPackingBoostrapKey$len(::rust::Vec<::PrivateFunctionalPackingBoostrapKey> const *ptr) noexcept;
::std::size_t cxxbridge1$rust_vec$PrivateFunctionalPackingBoostrapKey$capacity(::rust::Vec<::PrivateFunctionalPackingBoostrapKey> const *ptr) noexcept;
::PrivateFunctionalPackingBoostrapKey const *cxxbridge1$rust_vec$PrivateFunctionalPackingBoostrapKey$data(::rust::Vec<::PrivateFunctionalPackingBoostrapKey> const *ptr) noexcept;
void cxxbridge1$rust_vec$PrivateFunctionalPackingBoostrapKey$reserve_total(::rust::Vec<::PrivateFunctionalPackingBoostrapKey> *ptr, ::std::size_t new_cap) noexcept;
void cxxbridge1$rust_vec$PrivateFunctionalPackingBoostrapKey$set_len(::rust::Vec<::PrivateFunctionalPackingBoostrapKey> *ptr, ::std::size_t len) noexcept;
void cxxbridge1$rust_vec$PrivateFunctionalPackingBoostrapKey$truncate(::rust::Vec<::PrivateFunctionalPackingBoostrapKey> *ptr, ::std::size_t len) noexcept;

void cxxbridge1$rust_vec$concrete_optimizer$dag$InstructionKeys$new(::rust::Vec<::concrete_optimizer::dag::InstructionKeys> const *ptr) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$InstructionKeys$drop(::rust::Vec<::concrete_optimizer::dag::InstructionKeys> *ptr) noexcept;
::std::size_t cxxbridge1$rust_vec$concrete_optimizer$dag$InstructionKeys$len(::rust::Vec<::concrete_optimizer::dag::InstructionKeys> const *ptr) noexcept;
::std::size_t cxxbridge1$rust_vec$concrete_optimizer$dag$InstructionKeys$capacity(::rust::Vec<::concrete_optimizer::dag::InstructionKeys> const *ptr) noexcept;
::concrete_optimizer::dag::InstructionKeys const *cxxbridge1$rust_vec$concrete_optimizer$dag$InstructionKeys$data(::rust::Vec<::concrete_optimizer::dag::InstructionKeys> const *ptr) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$InstructionKeys$reserve_total(::rust::Vec<::concrete_optimizer::dag::InstructionKeys> *ptr, ::std::size_t new_cap) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$InstructionKeys$set_len(::rust::Vec<::concrete_optimizer::dag::InstructionKeys> *ptr, ::std::size_t len) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$dag$InstructionKeys$truncate(::rust::Vec<::concrete_optimizer::dag::InstructionKeys> *ptr, ::std::size_t len) noexcept;

void cxxbridge1$rust_vec$concrete_optimizer$restriction$LweSecretKeyInfo$new(::rust::Vec<::concrete_optimizer::restriction::LweSecretKeyInfo> const *ptr) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$restriction$LweSecretKeyInfo$drop(::rust::Vec<::concrete_optimizer::restriction::LweSecretKeyInfo> *ptr) noexcept;
::std::size_t cxxbridge1$rust_vec$concrete_optimizer$restriction$LweSecretKeyInfo$len(::rust::Vec<::concrete_optimizer::restriction::LweSecretKeyInfo> const *ptr) noexcept;
::std::size_t cxxbridge1$rust_vec$concrete_optimizer$restriction$LweSecretKeyInfo$capacity(::rust::Vec<::concrete_optimizer::restriction::LweSecretKeyInfo> const *ptr) noexcept;
::concrete_optimizer::restriction::LweSecretKeyInfo const *cxxbridge1$rust_vec$concrete_optimizer$restriction$LweSecretKeyInfo$data(::rust::Vec<::concrete_optimizer::restriction::LweSecretKeyInfo> const *ptr) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$restriction$LweSecretKeyInfo$reserve_total(::rust::Vec<::concrete_optimizer::restriction::LweSecretKeyInfo> *ptr, ::std::size_t new_cap) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$restriction$LweSecretKeyInfo$set_len(::rust::Vec<::concrete_optimizer::restriction::LweSecretKeyInfo> *ptr, ::std::size_t len) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$restriction$LweSecretKeyInfo$truncate(::rust::Vec<::concrete_optimizer::restriction::LweSecretKeyInfo> *ptr, ::std::size_t len) noexcept;

void cxxbridge1$rust_vec$concrete_optimizer$restriction$LweBootstrapKeyInfo$new(::rust::Vec<::concrete_optimizer::restriction::LweBootstrapKeyInfo> const *ptr) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$restriction$LweBootstrapKeyInfo$drop(::rust::Vec<::concrete_optimizer::restriction::LweBootstrapKeyInfo> *ptr) noexcept;
::std::size_t cxxbridge1$rust_vec$concrete_optimizer$restriction$LweBootstrapKeyInfo$len(::rust::Vec<::concrete_optimizer::restriction::LweBootstrapKeyInfo> const *ptr) noexcept;
::std::size_t cxxbridge1$rust_vec$concrete_optimizer$restriction$LweBootstrapKeyInfo$capacity(::rust::Vec<::concrete_optimizer::restriction::LweBootstrapKeyInfo> const *ptr) noexcept;
::concrete_optimizer::restriction::LweBootstrapKeyInfo const *cxxbridge1$rust_vec$concrete_optimizer$restriction$LweBootstrapKeyInfo$data(::rust::Vec<::concrete_optimizer::restriction::LweBootstrapKeyInfo> const *ptr) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$restriction$LweBootstrapKeyInfo$reserve_total(::rust::Vec<::concrete_optimizer::restriction::LweBootstrapKeyInfo> *ptr, ::std::size_t new_cap) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$restriction$LweBootstrapKeyInfo$set_len(::rust::Vec<::concrete_optimizer::restriction::LweBootstrapKeyInfo> *ptr, ::std::size_t len) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$restriction$LweBootstrapKeyInfo$truncate(::rust::Vec<::concrete_optimizer::restriction::LweBootstrapKeyInfo> *ptr, ::std::size_t len) noexcept;

void cxxbridge1$rust_vec$concrete_optimizer$restriction$LweKeyswitchKeyInfo$new(::rust::Vec<::concrete_optimizer::restriction::LweKeyswitchKeyInfo> const *ptr) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$restriction$LweKeyswitchKeyInfo$drop(::rust::Vec<::concrete_optimizer::restriction::LweKeyswitchKeyInfo> *ptr) noexcept;
::std::size_t cxxbridge1$rust_vec$concrete_optimizer$restriction$LweKeyswitchKeyInfo$len(::rust::Vec<::concrete_optimizer::restriction::LweKeyswitchKeyInfo> const *ptr) noexcept;
::std::size_t cxxbridge1$rust_vec$concrete_optimizer$restriction$LweKeyswitchKeyInfo$capacity(::rust::Vec<::concrete_optimizer::restriction::LweKeyswitchKeyInfo> const *ptr) noexcept;
::concrete_optimizer::restriction::LweKeyswitchKeyInfo const *cxxbridge1$rust_vec$concrete_optimizer$restriction$LweKeyswitchKeyInfo$data(::rust::Vec<::concrete_optimizer::restriction::LweKeyswitchKeyInfo> const *ptr) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$restriction$LweKeyswitchKeyInfo$reserve_total(::rust::Vec<::concrete_optimizer::restriction::LweKeyswitchKeyInfo> *ptr, ::std::size_t new_cap) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$restriction$LweKeyswitchKeyInfo$set_len(::rust::Vec<::concrete_optimizer::restriction::LweKeyswitchKeyInfo> *ptr, ::std::size_t len) noexcept;
void cxxbridge1$rust_vec$concrete_optimizer$restriction$LweKeyswitchKeyInfo$truncate(::rust::Vec<::concrete_optimizer::restriction::LweKeyswitchKeyInfo> *ptr, ::std::size_t len) noexcept;
} // extern "C"

namespace rust {
inline namespace cxxbridge1 {
template <>
::concrete_optimizer::Location *Box<::concrete_optimizer::Location>::allocation::alloc() noexcept {
  return cxxbridge1$box$concrete_optimizer$Location$alloc();
}
template <>
void Box<::concrete_optimizer::Location>::allocation::dealloc(::concrete_optimizer::Location *ptr) noexcept {
  cxxbridge1$box$concrete_optimizer$Location$dealloc(ptr);
}
template <>
void Box<::concrete_optimizer::Location>::drop() noexcept {
  cxxbridge1$box$concrete_optimizer$Location$drop(this);
}
template <>
Vec<::concrete_optimizer::utils::PartitionDefinition>::Vec() noexcept {
  cxxbridge1$rust_vec$concrete_optimizer$utils$PartitionDefinition$new(this);
}
template <>
void Vec<::concrete_optimizer::utils::PartitionDefinition>::drop() noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$utils$PartitionDefinition$drop(this);
}
template <>
::std::size_t Vec<::concrete_optimizer::utils::PartitionDefinition>::size() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$utils$PartitionDefinition$len(this);
}
template <>
::std::size_t Vec<::concrete_optimizer::utils::PartitionDefinition>::capacity() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$utils$PartitionDefinition$capacity(this);
}
template <>
::concrete_optimizer::utils::PartitionDefinition const *Vec<::concrete_optimizer::utils::PartitionDefinition>::data() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$utils$PartitionDefinition$data(this);
}
template <>
void Vec<::concrete_optimizer::utils::PartitionDefinition>::reserve_total(::std::size_t new_cap) noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$utils$PartitionDefinition$reserve_total(this, new_cap);
}
template <>
void Vec<::concrete_optimizer::utils::PartitionDefinition>::set_len(::std::size_t len) noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$utils$PartitionDefinition$set_len(this, len);
}
template <>
void Vec<::concrete_optimizer::utils::PartitionDefinition>::truncate(::std::size_t len) {
  return cxxbridge1$rust_vec$concrete_optimizer$utils$PartitionDefinition$truncate(this, len);
}
template <>
::concrete_optimizer::ExternalPartition *Box<::concrete_optimizer::ExternalPartition>::allocation::alloc() noexcept {
  return cxxbridge1$box$concrete_optimizer$ExternalPartition$alloc();
}
template <>
void Box<::concrete_optimizer::ExternalPartition>::allocation::dealloc(::concrete_optimizer::ExternalPartition *ptr) noexcept {
  cxxbridge1$box$concrete_optimizer$ExternalPartition$dealloc(ptr);
}
template <>
void Box<::concrete_optimizer::ExternalPartition>::drop() noexcept {
  cxxbridge1$box$concrete_optimizer$ExternalPartition$drop(this);
}
template <>
::concrete_optimizer::Dag *Box<::concrete_optimizer::Dag>::allocation::alloc() noexcept {
  return cxxbridge1$box$concrete_optimizer$Dag$alloc();
}
template <>
void Box<::concrete_optimizer::Dag>::allocation::dealloc(::concrete_optimizer::Dag *ptr) noexcept {
  cxxbridge1$box$concrete_optimizer$Dag$dealloc(ptr);
}
template <>
void Box<::concrete_optimizer::Dag>::drop() noexcept {
  cxxbridge1$box$concrete_optimizer$Dag$drop(this);
}
template <>
::concrete_optimizer::DagBuilder *Box<::concrete_optimizer::DagBuilder>::allocation::alloc() noexcept {
  return cxxbridge1$box$concrete_optimizer$DagBuilder$alloc();
}
template <>
void Box<::concrete_optimizer::DagBuilder>::allocation::dealloc(::concrete_optimizer::DagBuilder *ptr) noexcept {
  cxxbridge1$box$concrete_optimizer$DagBuilder$dealloc(ptr);
}
template <>
void Box<::concrete_optimizer::DagBuilder>::drop() noexcept {
  cxxbridge1$box$concrete_optimizer$DagBuilder$drop(this);
}
template <>
::concrete_optimizer::Weights *Box<::concrete_optimizer::Weights>::allocation::alloc() noexcept {
  return cxxbridge1$box$concrete_optimizer$Weights$alloc();
}
template <>
void Box<::concrete_optimizer::Weights>::allocation::dealloc(::concrete_optimizer::Weights *ptr) noexcept {
  cxxbridge1$box$concrete_optimizer$Weights$dealloc(ptr);
}
template <>
void Box<::concrete_optimizer::Weights>::drop() noexcept {
  cxxbridge1$box$concrete_optimizer$Weights$drop(this);
}
template <>
Vec<::concrete_optimizer::dag::OperatorIndex>::Vec() noexcept {
  cxxbridge1$rust_vec$concrete_optimizer$dag$OperatorIndex$new(this);
}
template <>
void Vec<::concrete_optimizer::dag::OperatorIndex>::drop() noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$OperatorIndex$drop(this);
}
template <>
::std::size_t Vec<::concrete_optimizer::dag::OperatorIndex>::size() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$OperatorIndex$len(this);
}
template <>
::std::size_t Vec<::concrete_optimizer::dag::OperatorIndex>::capacity() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$OperatorIndex$capacity(this);
}
template <>
::concrete_optimizer::dag::OperatorIndex const *Vec<::concrete_optimizer::dag::OperatorIndex>::data() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$OperatorIndex$data(this);
}
template <>
void Vec<::concrete_optimizer::dag::OperatorIndex>::reserve_total(::std::size_t new_cap) noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$OperatorIndex$reserve_total(this, new_cap);
}
template <>
void Vec<::concrete_optimizer::dag::OperatorIndex>::set_len(::std::size_t len) noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$OperatorIndex$set_len(this, len);
}
template <>
void Vec<::concrete_optimizer::dag::OperatorIndex>::truncate(::std::size_t len) {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$OperatorIndex$truncate(this, len);
}
template <>
Vec<::concrete_optimizer::dag::SecretLweKey>::Vec() noexcept {
  cxxbridge1$rust_vec$concrete_optimizer$dag$SecretLweKey$new(this);
}
template <>
void Vec<::concrete_optimizer::dag::SecretLweKey>::drop() noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$SecretLweKey$drop(this);
}
template <>
::std::size_t Vec<::concrete_optimizer::dag::SecretLweKey>::size() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$SecretLweKey$len(this);
}
template <>
::std::size_t Vec<::concrete_optimizer::dag::SecretLweKey>::capacity() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$SecretLweKey$capacity(this);
}
template <>
::concrete_optimizer::dag::SecretLweKey const *Vec<::concrete_optimizer::dag::SecretLweKey>::data() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$SecretLweKey$data(this);
}
template <>
void Vec<::concrete_optimizer::dag::SecretLweKey>::reserve_total(::std::size_t new_cap) noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$SecretLweKey$reserve_total(this, new_cap);
}
template <>
void Vec<::concrete_optimizer::dag::SecretLweKey>::set_len(::std::size_t len) noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$SecretLweKey$set_len(this, len);
}
template <>
void Vec<::concrete_optimizer::dag::SecretLweKey>::truncate(::std::size_t len) {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$SecretLweKey$truncate(this, len);
}
template <>
Vec<::concrete_optimizer::dag::KeySwitchKey>::Vec() noexcept {
  cxxbridge1$rust_vec$concrete_optimizer$dag$KeySwitchKey$new(this);
}
template <>
void Vec<::concrete_optimizer::dag::KeySwitchKey>::drop() noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$KeySwitchKey$drop(this);
}
template <>
::std::size_t Vec<::concrete_optimizer::dag::KeySwitchKey>::size() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$KeySwitchKey$len(this);
}
template <>
::std::size_t Vec<::concrete_optimizer::dag::KeySwitchKey>::capacity() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$KeySwitchKey$capacity(this);
}
template <>
::concrete_optimizer::dag::KeySwitchKey const *Vec<::concrete_optimizer::dag::KeySwitchKey>::data() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$KeySwitchKey$data(this);
}
template <>
void Vec<::concrete_optimizer::dag::KeySwitchKey>::reserve_total(::std::size_t new_cap) noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$KeySwitchKey$reserve_total(this, new_cap);
}
template <>
void Vec<::concrete_optimizer::dag::KeySwitchKey>::set_len(::std::size_t len) noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$KeySwitchKey$set_len(this, len);
}
template <>
void Vec<::concrete_optimizer::dag::KeySwitchKey>::truncate(::std::size_t len) {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$KeySwitchKey$truncate(this, len);
}
template <>
Vec<::concrete_optimizer::dag::BootstrapKey>::Vec() noexcept {
  cxxbridge1$rust_vec$concrete_optimizer$dag$BootstrapKey$new(this);
}
template <>
void Vec<::concrete_optimizer::dag::BootstrapKey>::drop() noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$BootstrapKey$drop(this);
}
template <>
::std::size_t Vec<::concrete_optimizer::dag::BootstrapKey>::size() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$BootstrapKey$len(this);
}
template <>
::std::size_t Vec<::concrete_optimizer::dag::BootstrapKey>::capacity() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$BootstrapKey$capacity(this);
}
template <>
::concrete_optimizer::dag::BootstrapKey const *Vec<::concrete_optimizer::dag::BootstrapKey>::data() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$BootstrapKey$data(this);
}
template <>
void Vec<::concrete_optimizer::dag::BootstrapKey>::reserve_total(::std::size_t new_cap) noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$BootstrapKey$reserve_total(this, new_cap);
}
template <>
void Vec<::concrete_optimizer::dag::BootstrapKey>::set_len(::std::size_t len) noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$BootstrapKey$set_len(this, len);
}
template <>
void Vec<::concrete_optimizer::dag::BootstrapKey>::truncate(::std::size_t len) {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$BootstrapKey$truncate(this, len);
}
template <>
Vec<::concrete_optimizer::dag::ConversionKeySwitchKey>::Vec() noexcept {
  cxxbridge1$rust_vec$concrete_optimizer$dag$ConversionKeySwitchKey$new(this);
}
template <>
void Vec<::concrete_optimizer::dag::ConversionKeySwitchKey>::drop() noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$ConversionKeySwitchKey$drop(this);
}
template <>
::std::size_t Vec<::concrete_optimizer::dag::ConversionKeySwitchKey>::size() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$ConversionKeySwitchKey$len(this);
}
template <>
::std::size_t Vec<::concrete_optimizer::dag::ConversionKeySwitchKey>::capacity() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$ConversionKeySwitchKey$capacity(this);
}
template <>
::concrete_optimizer::dag::ConversionKeySwitchKey const *Vec<::concrete_optimizer::dag::ConversionKeySwitchKey>::data() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$ConversionKeySwitchKey$data(this);
}
template <>
void Vec<::concrete_optimizer::dag::ConversionKeySwitchKey>::reserve_total(::std::size_t new_cap) noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$ConversionKeySwitchKey$reserve_total(this, new_cap);
}
template <>
void Vec<::concrete_optimizer::dag::ConversionKeySwitchKey>::set_len(::std::size_t len) noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$ConversionKeySwitchKey$set_len(this, len);
}
template <>
void Vec<::concrete_optimizer::dag::ConversionKeySwitchKey>::truncate(::std::size_t len) {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$ConversionKeySwitchKey$truncate(this, len);
}
template <>
Vec<::concrete_optimizer::dag::CircuitBoostrapKey>::Vec() noexcept {
  cxxbridge1$rust_vec$concrete_optimizer$dag$CircuitBoostrapKey$new(this);
}
template <>
void Vec<::concrete_optimizer::dag::CircuitBoostrapKey>::drop() noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$CircuitBoostrapKey$drop(this);
}
template <>
::std::size_t Vec<::concrete_optimizer::dag::CircuitBoostrapKey>::size() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$CircuitBoostrapKey$len(this);
}
template <>
::std::size_t Vec<::concrete_optimizer::dag::CircuitBoostrapKey>::capacity() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$CircuitBoostrapKey$capacity(this);
}
template <>
::concrete_optimizer::dag::CircuitBoostrapKey const *Vec<::concrete_optimizer::dag::CircuitBoostrapKey>::data() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$CircuitBoostrapKey$data(this);
}
template <>
void Vec<::concrete_optimizer::dag::CircuitBoostrapKey>::reserve_total(::std::size_t new_cap) noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$CircuitBoostrapKey$reserve_total(this, new_cap);
}
template <>
void Vec<::concrete_optimizer::dag::CircuitBoostrapKey>::set_len(::std::size_t len) noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$CircuitBoostrapKey$set_len(this, len);
}
template <>
void Vec<::concrete_optimizer::dag::CircuitBoostrapKey>::truncate(::std::size_t len) {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$CircuitBoostrapKey$truncate(this, len);
}
template <>
Vec<::PrivateFunctionalPackingBoostrapKey>::Vec() noexcept {
  cxxbridge1$rust_vec$PrivateFunctionalPackingBoostrapKey$new(this);
}
template <>
void Vec<::PrivateFunctionalPackingBoostrapKey>::drop() noexcept {
  return cxxbridge1$rust_vec$PrivateFunctionalPackingBoostrapKey$drop(this);
}
template <>
::std::size_t Vec<::PrivateFunctionalPackingBoostrapKey>::size() const noexcept {
  return cxxbridge1$rust_vec$PrivateFunctionalPackingBoostrapKey$len(this);
}
template <>
::std::size_t Vec<::PrivateFunctionalPackingBoostrapKey>::capacity() const noexcept {
  return cxxbridge1$rust_vec$PrivateFunctionalPackingBoostrapKey$capacity(this);
}
template <>
::PrivateFunctionalPackingBoostrapKey const *Vec<::PrivateFunctionalPackingBoostrapKey>::data() const noexcept {
  return cxxbridge1$rust_vec$PrivateFunctionalPackingBoostrapKey$data(this);
}
template <>
void Vec<::PrivateFunctionalPackingBoostrapKey>::reserve_total(::std::size_t new_cap) noexcept {
  return cxxbridge1$rust_vec$PrivateFunctionalPackingBoostrapKey$reserve_total(this, new_cap);
}
template <>
void Vec<::PrivateFunctionalPackingBoostrapKey>::set_len(::std::size_t len) noexcept {
  return cxxbridge1$rust_vec$PrivateFunctionalPackingBoostrapKey$set_len(this, len);
}
template <>
void Vec<::PrivateFunctionalPackingBoostrapKey>::truncate(::std::size_t len) {
  return cxxbridge1$rust_vec$PrivateFunctionalPackingBoostrapKey$truncate(this, len);
}
template <>
Vec<::concrete_optimizer::dag::InstructionKeys>::Vec() noexcept {
  cxxbridge1$rust_vec$concrete_optimizer$dag$InstructionKeys$new(this);
}
template <>
void Vec<::concrete_optimizer::dag::InstructionKeys>::drop() noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$InstructionKeys$drop(this);
}
template <>
::std::size_t Vec<::concrete_optimizer::dag::InstructionKeys>::size() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$InstructionKeys$len(this);
}
template <>
::std::size_t Vec<::concrete_optimizer::dag::InstructionKeys>::capacity() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$InstructionKeys$capacity(this);
}
template <>
::concrete_optimizer::dag::InstructionKeys const *Vec<::concrete_optimizer::dag::InstructionKeys>::data() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$InstructionKeys$data(this);
}
template <>
void Vec<::concrete_optimizer::dag::InstructionKeys>::reserve_total(::std::size_t new_cap) noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$InstructionKeys$reserve_total(this, new_cap);
}
template <>
void Vec<::concrete_optimizer::dag::InstructionKeys>::set_len(::std::size_t len) noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$InstructionKeys$set_len(this, len);
}
template <>
void Vec<::concrete_optimizer::dag::InstructionKeys>::truncate(::std::size_t len) {
  return cxxbridge1$rust_vec$concrete_optimizer$dag$InstructionKeys$truncate(this, len);
}
template <>
Vec<::concrete_optimizer::restriction::LweSecretKeyInfo>::Vec() noexcept {
  cxxbridge1$rust_vec$concrete_optimizer$restriction$LweSecretKeyInfo$new(this);
}
template <>
void Vec<::concrete_optimizer::restriction::LweSecretKeyInfo>::drop() noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$restriction$LweSecretKeyInfo$drop(this);
}
template <>
::std::size_t Vec<::concrete_optimizer::restriction::LweSecretKeyInfo>::size() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$restriction$LweSecretKeyInfo$len(this);
}
template <>
::std::size_t Vec<::concrete_optimizer::restriction::LweSecretKeyInfo>::capacity() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$restriction$LweSecretKeyInfo$capacity(this);
}
template <>
::concrete_optimizer::restriction::LweSecretKeyInfo const *Vec<::concrete_optimizer::restriction::LweSecretKeyInfo>::data() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$restriction$LweSecretKeyInfo$data(this);
}
template <>
void Vec<::concrete_optimizer::restriction::LweSecretKeyInfo>::reserve_total(::std::size_t new_cap) noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$restriction$LweSecretKeyInfo$reserve_total(this, new_cap);
}
template <>
void Vec<::concrete_optimizer::restriction::LweSecretKeyInfo>::set_len(::std::size_t len) noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$restriction$LweSecretKeyInfo$set_len(this, len);
}
template <>
void Vec<::concrete_optimizer::restriction::LweSecretKeyInfo>::truncate(::std::size_t len) {
  return cxxbridge1$rust_vec$concrete_optimizer$restriction$LweSecretKeyInfo$truncate(this, len);
}
template <>
Vec<::concrete_optimizer::restriction::LweBootstrapKeyInfo>::Vec() noexcept {
  cxxbridge1$rust_vec$concrete_optimizer$restriction$LweBootstrapKeyInfo$new(this);
}
template <>
void Vec<::concrete_optimizer::restriction::LweBootstrapKeyInfo>::drop() noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$restriction$LweBootstrapKeyInfo$drop(this);
}
template <>
::std::size_t Vec<::concrete_optimizer::restriction::LweBootstrapKeyInfo>::size() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$restriction$LweBootstrapKeyInfo$len(this);
}
template <>
::std::size_t Vec<::concrete_optimizer::restriction::LweBootstrapKeyInfo>::capacity() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$restriction$LweBootstrapKeyInfo$capacity(this);
}
template <>
::concrete_optimizer::restriction::LweBootstrapKeyInfo const *Vec<::concrete_optimizer::restriction::LweBootstrapKeyInfo>::data() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$restriction$LweBootstrapKeyInfo$data(this);
}
template <>
void Vec<::concrete_optimizer::restriction::LweBootstrapKeyInfo>::reserve_total(::std::size_t new_cap) noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$restriction$LweBootstrapKeyInfo$reserve_total(this, new_cap);
}
template <>
void Vec<::concrete_optimizer::restriction::LweBootstrapKeyInfo>::set_len(::std::size_t len) noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$restriction$LweBootstrapKeyInfo$set_len(this, len);
}
template <>
void Vec<::concrete_optimizer::restriction::LweBootstrapKeyInfo>::truncate(::std::size_t len) {
  return cxxbridge1$rust_vec$concrete_optimizer$restriction$LweBootstrapKeyInfo$truncate(this, len);
}
template <>
Vec<::concrete_optimizer::restriction::LweKeyswitchKeyInfo>::Vec() noexcept {
  cxxbridge1$rust_vec$concrete_optimizer$restriction$LweKeyswitchKeyInfo$new(this);
}
template <>
void Vec<::concrete_optimizer::restriction::LweKeyswitchKeyInfo>::drop() noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$restriction$LweKeyswitchKeyInfo$drop(this);
}
template <>
::std::size_t Vec<::concrete_optimizer::restriction::LweKeyswitchKeyInfo>::size() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$restriction$LweKeyswitchKeyInfo$len(this);
}
template <>
::std::size_t Vec<::concrete_optimizer::restriction::LweKeyswitchKeyInfo>::capacity() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$restriction$LweKeyswitchKeyInfo$capacity(this);
}
template <>
::concrete_optimizer::restriction::LweKeyswitchKeyInfo const *Vec<::concrete_optimizer::restriction::LweKeyswitchKeyInfo>::data() const noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$restriction$LweKeyswitchKeyInfo$data(this);
}
template <>
void Vec<::concrete_optimizer::restriction::LweKeyswitchKeyInfo>::reserve_total(::std::size_t new_cap) noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$restriction$LweKeyswitchKeyInfo$reserve_total(this, new_cap);
}
template <>
void Vec<::concrete_optimizer::restriction::LweKeyswitchKeyInfo>::set_len(::std::size_t len) noexcept {
  return cxxbridge1$rust_vec$concrete_optimizer$restriction$LweKeyswitchKeyInfo$set_len(this, len);
}
template <>
void Vec<::concrete_optimizer::restriction::LweKeyswitchKeyInfo>::truncate(::std::size_t len) {
  return cxxbridge1$rust_vec$concrete_optimizer$restriction$LweKeyswitchKeyInfo$truncate(this, len);
}
} // namespace cxxbridge1
} // namespace rust
