/* Include the default amount of outcome
(C) 2018-2021 Niall Douglas <http://www.nedproductions.biz/> (4 commits)
File Created: Mar 2018


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#if !OUTCOME_ENABLE_CXX_MODULES || !0 || defined(GENERATING_OUTCOME_MODULE_INTERFACE) || OUTCOME_DISABLE_CXX_MODULES
/* Tells C++ coroutines about Outcome's result
(C) 2019 Niall Douglas <http://www.nedproductions.biz/> (12 commits)
File Created: Oct 2019


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_COROUTINE_SUPPORT_HPP
#define OUTCOME_COROUTINE_SUPPORT_HPP
/* Configure Outcome with QuickCppLib
(C) 2015-2021 Niall Douglas <http://www.nedproductions.biz/> (24 commits)
File Created: August 2015


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_V2_CONFIG_HPP
#define OUTCOME_V2_CONFIG_HPP
/* Sets Outcome version
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (4 commits)


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
/*! AWAITING HUGO JSON CONVERSION TOOL */
#define OUTCOME_VERSION_MAJOR 2
/*! AWAITING HUGO JSON CONVERSION TOOL */
#define OUTCOME_VERSION_MINOR 2
/*! AWAITING HUGO JSON CONVERSION TOOL */
#define OUTCOME_VERSION_PATCH 0
/*! AWAITING HUGO JSON CONVERSION TOOL */
#define OUTCOME_VERSION_REVISION 0 // Revision version for cmake and DLL version stamping
/*! AWAITING HUGO JSON CONVERSION TOOL */
#ifndef OUTCOME_DISABLE_ABI_PERMUTATION
#define OUTCOME_UNSTABLE_VERSION
#endif
// Pull in detection of __MINGW64_VERSION_MAJOR
#if defined(__MINGW32__) && !0
#include <_mingw.h>
#endif
/* Configure QuickCppLib
(C) 2016-2021 Niall Douglas <http://www.nedproductions.biz/> (8 commits)


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef QUICKCPPLIB_CONFIG_HPP
#define QUICKCPPLIB_CONFIG_HPP
/* Provides SG-10 feature checking for all C++ compilers
(C) 2014-2017 Niall Douglas <http://www.nedproductions.biz/> (13 commits)
File Created: Nov 2014


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef QUICKCPPLIB_HAS_FEATURE_H
#define QUICKCPPLIB_HAS_FEATURE_H
#if __cplusplus >= 201103
// Some of these macros ended up getting removed by ISO standards,
// they are prefixed with ////
////#if !defined(__cpp_alignas)
////#define __cpp_alignas 190000
////#endif
////#if !defined(__cpp_default_function_template_args)
////#define __cpp_default_function_template_args 190000
////#endif
////#if !defined(__cpp_defaulted_functions)
////#define __cpp_defaulted_functions 190000
////#endif
////#if !defined(__cpp_deleted_functions)
////#define __cpp_deleted_functions 190000
////#endif
////#if !defined(__cpp_generalized_initializers)
////#define __cpp_generalized_initializers 190000
////#endif
////#if !defined(__cpp_implicit_moves)
////#define __cpp_implicit_moves 190000
////#endif
////#if !defined(__cpp_inline_namespaces)
////#define __cpp_inline_namespaces 190000
////#endif
////#if !defined(__cpp_local_type_template_args)
////#define __cpp_local_type_template_args 190000
////#endif
////#if !defined(__cpp_noexcept)
////#define __cpp_noexcept 190000
////#endif
////#if !defined(__cpp_nonstatic_member_init)
////#define __cpp_nonstatic_member_init 190000
////#endif
////#if !defined(__cpp_nullptr)
////#define __cpp_nullptr 190000
////#endif
////#if !defined(__cpp_override_control)
////#define __cpp_override_control 190000
////#endif
////#if !defined(__cpp_thread_local)
////#define __cpp_thread_local 190000
////#endif
////#if !defined(__cpp_auto_type)
////#define __cpp_auto_type 190000
////#endif
////#if !defined(__cpp_strong_enums)
////#define __cpp_strong_enums 190000
////#endif
////#if !defined(__cpp_trailing_return)
////#define __cpp_trailing_return 190000
////#endif
////#if !defined(__cpp_unrestricted_unions)
////#define __cpp_unrestricted_unions 190000
////#endif
#if !defined(__cpp_alias_templates)
#define __cpp_alias_templates 190000
#endif
#if !defined(__cpp_attributes)
#define __cpp_attributes 190000
#endif
#if !defined(__cpp_constexpr)
#if __cplusplus >= 201402
#define __cpp_constexpr 201304 // relaxed constexpr
#else
#define __cpp_constexpr 190000
#endif
#endif
#if !defined(__cpp_decltype)
#define __cpp_decltype 190000
#endif
#if !defined(__cpp_delegating_constructors)
#define __cpp_delegating_constructors 190000
#endif
#if !defined(__cpp_explicit_conversion) //// renamed from __cpp_explicit_conversions
#define __cpp_explicit_conversion 190000
#endif
#if !defined(__cpp_inheriting_constructors)
#define __cpp_inheriting_constructors 190000
#endif
#if !defined(__cpp_initializer_lists) //// NEW
#define __cpp_initializer_lists 190000
#endif
#if !defined(__cpp_lambdas)
#define __cpp_lambdas 190000
#endif
#if !defined(__cpp_nsdmi)
#define __cpp_nsdmi 190000 //// NEW
#endif
#if !defined(__cpp_range_based_for) //// renamed from __cpp_range_for
#define __cpp_range_based_for 190000
#endif
#if !defined(__cpp_raw_strings)
#define __cpp_raw_strings 190000
#endif
#if !defined(__cpp_ref_qualifiers) //// renamed from __cpp_reference_qualified_functions
#define __cpp_ref_qualifiers 190000
#endif
#if !defined(__cpp_rvalue_references)
#define __cpp_rvalue_references 190000
#endif
#if !defined(__cpp_static_assert)
#define __cpp_static_assert 190000
#endif
#if !defined(__cpp_unicode_characters) //// NEW
#define __cpp_unicode_characters 190000
#endif
#if !defined(__cpp_unicode_literals)
#define __cpp_unicode_literals 190000
#endif
#if !defined(__cpp_user_defined_literals)
#define __cpp_user_defined_literals 190000
#endif
#if !defined(__cpp_variadic_templates)
#define __cpp_variadic_templates 190000
#endif
#endif
#if __cplusplus >= 201402
// Some of these macros ended up getting removed by ISO standards,
// they are prefixed with ////
////#if !defined(__cpp_contextual_conversions)
////#define __cpp_contextual_conversions 190000
////#endif
////#if !defined(__cpp_digit_separators)
////#define __cpp_digit_separators 190000
////#endif
////#if !defined(__cpp_relaxed_constexpr)
////#define __cpp_relaxed_constexpr 190000
////#endif
////#if !defined(__cpp_runtime_arrays)
////# define __cpp_runtime_arrays 190000
////#endif
#if !defined(__cpp_aggregate_nsdmi)
#define __cpp_aggregate_nsdmi 190000
#endif
#if !defined(__cpp_binary_literals)
#define __cpp_binary_literals 190000
#endif
#if !defined(__cpp_decltype_auto)
#define __cpp_decltype_auto 190000
#endif
#if !defined(__cpp_generic_lambdas)
#define __cpp_generic_lambdas 190000
#endif
#if !defined(__cpp_init_captures)
#define __cpp_init_captures 190000
#endif
#if !defined(__cpp_return_type_deduction)
#define __cpp_return_type_deduction 190000
#endif
#if !defined(__cpp_sized_deallocation)
#define __cpp_sized_deallocation 190000
#endif
#if !defined(__cpp_variable_templates)
#define __cpp_variable_templates 190000
#endif
#endif
// VS2010: _MSC_VER=1600
// VS2012: _MSC_VER=1700
// VS2013: _MSC_VER=1800
// VS2015: _MSC_VER=1900
// VS2017: _MSC_VER=1910
#if defined(_MSC_VER) && !defined(__clang__)
#if !defined(__cpp_exceptions) && defined(_CPPUNWIND)
#define __cpp_exceptions 190000
#endif
#if !defined(__cpp_rtti) && defined(_CPPRTTI)
#define __cpp_rtti 190000
#endif
// C++ 11
#if !defined(__cpp_alias_templates) && _MSC_VER >= 1800
#define __cpp_alias_templates 190000
#endif
#if !defined(__cpp_attributes)
#define __cpp_attributes 190000
#endif
#if !defined(__cpp_constexpr) && _MSC_FULL_VER >= 190023506 /* VS2015 */
#define __cpp_constexpr 190000
#endif
#if !defined(__cpp_decltype) && _MSC_VER >= 1600
#define __cpp_decltype 190000
#endif
#if !defined(__cpp_delegating_constructors) && _MSC_VER >= 1800
#define __cpp_delegating_constructors 190000
#endif
#if !defined(__cpp_explicit_conversion) && _MSC_VER >= 1800
#define __cpp_explicit_conversion 190000
#endif
#if !defined(__cpp_inheriting_constructors) && _MSC_VER >= 1900
#define __cpp_inheriting_constructors 190000
#endif
#if !defined(__cpp_initializer_lists) && _MSC_VER >= 1900
#define __cpp_initializer_lists 190000
#endif
#if !defined(__cpp_lambdas) && _MSC_VER >= 1600
#define __cpp_lambdas 190000
#endif
#if !defined(__cpp_nsdmi) && _MSC_VER >= 1900
#define __cpp_nsdmi 190000
#endif
#if !defined(__cpp_range_based_for) && _MSC_VER >= 1700
#define __cpp_range_based_for 190000
#endif
#if !defined(__cpp_raw_strings) && _MSC_VER >= 1800
#define __cpp_raw_strings 190000
#endif
#if !defined(__cpp_ref_qualifiers) && _MSC_VER >= 1900
#define __cpp_ref_qualifiers 190000
#endif
#if !defined(__cpp_rvalue_references) && _MSC_VER >= 1600
#define __cpp_rvalue_references 190000
#endif
#if !defined(__cpp_static_assert) && _MSC_VER >= 1600
#define __cpp_static_assert 190000
#endif
//#if !defined(__cpp_unicode_literals)
//# define __cpp_unicode_literals 190000
//#endif
#if !defined(__cpp_user_defined_literals) && _MSC_VER >= 1900
#define __cpp_user_defined_literals 190000
#endif
#if !defined(__cpp_variadic_templates) && _MSC_VER >= 1800
#define __cpp_variadic_templates 190000
#endif
// C++ 14
//#if !defined(__cpp_aggregate_nsdmi)
//#define __cpp_aggregate_nsdmi 190000
//#endif
#if !defined(__cpp_binary_literals) && _MSC_VER >= 1900
#define __cpp_binary_literals 190000
#endif
#if !defined(__cpp_decltype_auto) && _MSC_VER >= 1900
#define __cpp_decltype_auto 190000
#endif
#if !defined(__cpp_generic_lambdas) && _MSC_VER >= 1900
#define __cpp_generic_lambdas 190000
#endif
#if !defined(__cpp_init_captures) && _MSC_VER >= 1900
#define __cpp_init_captures 190000
#endif
#if !defined(__cpp_return_type_deduction) && _MSC_VER >= 1900
#define __cpp_return_type_deduction 190000
#endif
#if !defined(__cpp_sized_deallocation) && _MSC_VER >= 1900
#define __cpp_sized_deallocation 190000
#endif
#if !defined(__cpp_variable_templates) && _MSC_FULL_VER >= 190023506
#define __cpp_variable_templates 190000
#endif
#endif // _MSC_VER
// Much to my surprise, GCC's support of these is actually incomplete, so fill in the gaps
#if (defined(__GNUC__) && !defined(__clang__))
#define QUICKCPPLIB_GCC (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#if !defined(__cpp_exceptions) && defined(__EXCEPTIONS)
#define __cpp_exceptions 190000
#endif
#if !defined(__cpp_rtti) && defined(__GXX_RTTI)
#define __cpp_rtti 190000
#endif
// C++ 11
#if defined(__GXX_EXPERIMENTAL_CXX0X__)
#if !defined(__cpp_alias_templates) && (QUICKCPPLIB_GCC >= 40700)
#define __cpp_alias_templates 190000
#endif
#if !defined(__cpp_attributes) && (QUICKCPPLIB_GCC >= 40800)
#define __cpp_attributes 190000
#endif
#if !defined(__cpp_constexpr) && (QUICKCPPLIB_GCC >= 40600)
#define __cpp_constexpr 190000
#endif
#if !defined(__cpp_decltype) && (QUICKCPPLIB_GCC >= 40300)
#define __cpp_decltype 190000
#endif
#if !defined(__cpp_delegating_constructors) && (QUICKCPPLIB_GCC >= 40700)
#define __cpp_delegating_constructors 190000
#endif
#if !defined(__cpp_explicit_conversion) && (QUICKCPPLIB_GCC >= 40500)
#define __cpp_explicit_conversion 190000
#endif
#if !defined(__cpp_inheriting_constructors) && (QUICKCPPLIB_GCC >= 40800)
#define __cpp_inheriting_constructors 190000
#endif
#if !defined(__cpp_initializer_lists) && (QUICKCPPLIB_GCC >= 40800)
#define __cpp_initializer_lists 190000
#endif
#if !defined(__cpp_lambdas) && (QUICKCPPLIB_GCC >= 40500)
#define __cpp_lambdas 190000
#endif
#if !defined(__cpp_nsdmi) && (QUICKCPPLIB_GCC >= 40700)
#define __cpp_nsdmi 190000
#endif
#if !defined(__cpp_range_based_for) && (QUICKCPPLIB_GCC >= 40600)
#define __cpp_range_based_for 190000
#endif
#if !defined(__cpp_raw_strings) && (QUICKCPPLIB_GCC >= 40500)
#define __cpp_raw_strings 190000
#endif
#if !defined(__cpp_ref_qualifiers) && (QUICKCPPLIB_GCC >= 40801)
#define __cpp_ref_qualifiers 190000
#endif
// __cpp_rvalue_reference deviation
#if !defined(__cpp_rvalue_references) && defined(__cpp_rvalue_reference)
#define __cpp_rvalue_references __cpp_rvalue_reference
#endif
#if !defined(__cpp_static_assert) && (QUICKCPPLIB_GCC >= 40300)
#define __cpp_static_assert 190000
#endif
#if !defined(__cpp_unicode_characters) && (QUICKCPPLIB_GCC >= 40500)
#define __cpp_unicode_characters 190000
#endif
#if !defined(__cpp_unicode_literals) && (QUICKCPPLIB_GCC >= 40500)
#define __cpp_unicode_literals 190000
#endif
#if !defined(__cpp_user_defined_literals) && (QUICKCPPLIB_GCC >= 40700)
#define __cpp_user_defined_literals 190000
#endif
#if !defined(__cpp_variadic_templates) && (QUICKCPPLIB_GCC >= 40400)
#define __cpp_variadic_templates 190000
#endif
// C++ 14
// Every C++ 14 supporting GCC does the right thing here
#endif // __GXX_EXPERIMENTAL_CXX0X__
#endif // GCC
// clang deviates in some places from the present SG-10 draft, plus older
// clangs are quite incomplete
#if defined(__clang__)
#define QUICKCPPLIB_CLANG (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#if !defined(__cpp_exceptions) && (defined(__EXCEPTIONS) || defined(_CPPUNWIND))
#define __cpp_exceptions 190000
#endif
#if !defined(__cpp_rtti) && (defined(__GXX_RTTI) || defined(_CPPRTTI))
#define __cpp_rtti 190000
#endif
// C++ 11
#if defined(__GXX_EXPERIMENTAL_CXX0X__)
#if !defined(__cpp_alias_templates) && (QUICKCPPLIB_CLANG >= 30000)
#define __cpp_alias_templates 190000
#endif
#if !defined(__cpp_attributes) && (QUICKCPPLIB_CLANG >= 30300)
#define __cpp_attributes 190000
#endif
#if !defined(__cpp_constexpr) && (QUICKCPPLIB_CLANG >= 30100)
#define __cpp_constexpr 190000
#endif
#if !defined(__cpp_decltype) && (QUICKCPPLIB_CLANG >= 20900)
#define __cpp_decltype 190000
#endif
#if !defined(__cpp_delegating_constructors) && (QUICKCPPLIB_CLANG >= 30000)
#define __cpp_delegating_constructors 190000
#endif
#if !defined(__cpp_explicit_conversion) && (QUICKCPPLIB_CLANG >= 30000)
#define __cpp_explicit_conversion 190000
#endif
#if !defined(__cpp_inheriting_constructors) && (QUICKCPPLIB_CLANG >= 30300)
#define __cpp_inheriting_constructors 190000
#endif
#if !defined(__cpp_initializer_lists) && (QUICKCPPLIB_CLANG >= 30100)
#define __cpp_initializer_lists 190000
#endif
#if !defined(__cpp_lambdas) && (QUICKCPPLIB_CLANG >= 30100)
#define __cpp_lambdas 190000
#endif
#if !defined(__cpp_nsdmi) && (QUICKCPPLIB_CLANG >= 30000)
#define __cpp_nsdmi 190000
#endif
#if !defined(__cpp_range_based_for) && (QUICKCPPLIB_CLANG >= 30000)
#define __cpp_range_based_for 190000
#endif
// __cpp_raw_string_literals deviation
#if !defined(__cpp_raw_strings) && defined(__cpp_raw_string_literals)
#define __cpp_raw_strings __cpp_raw_string_literals
#endif
#if !defined(__cpp_raw_strings) && (QUICKCPPLIB_CLANG >= 30000)
#define __cpp_raw_strings 190000
#endif
#if !defined(__cpp_ref_qualifiers) && (QUICKCPPLIB_CLANG >= 20900)
#define __cpp_ref_qualifiers 190000
#endif
// __cpp_rvalue_reference deviation
#if !defined(__cpp_rvalue_references) && defined(__cpp_rvalue_reference)
#define __cpp_rvalue_references __cpp_rvalue_reference
#endif
#if !defined(__cpp_rvalue_references) && (QUICKCPPLIB_CLANG >= 20900)
#define __cpp_rvalue_references 190000
#endif
#if !defined(__cpp_static_assert) && (QUICKCPPLIB_CLANG >= 20900)
#define __cpp_static_assert 190000
#endif
#if !defined(__cpp_unicode_characters) && (QUICKCPPLIB_CLANG >= 30000)
#define __cpp_unicode_characters 190000
#endif
#if !defined(__cpp_unicode_literals) && (QUICKCPPLIB_CLANG >= 30000)
#define __cpp_unicode_literals 190000
#endif
// __cpp_user_literals deviation
#if !defined(__cpp_user_defined_literals) && defined(__cpp_user_literals)
#define __cpp_user_defined_literals __cpp_user_literals
#endif
#if !defined(__cpp_user_defined_literals) && (QUICKCPPLIB_CLANG >= 30100)
#define __cpp_user_defined_literals 190000
#endif
#if !defined(__cpp_variadic_templates) && (QUICKCPPLIB_CLANG >= 20900)
#define __cpp_variadic_templates 190000
#endif
// C++ 14
// Every C++ 14 supporting clang does the right thing here
#endif // __GXX_EXPERIMENTAL_CXX0X__
#endif // clang
#endif
#ifndef QUICKCPPLIB_DISABLE_ABI_PERMUTATION
// Note the second line of this file must ALWAYS be the git SHA, third line ALWAYS the git SHA update time
#define QUICKCPPLIB_PREVIOUS_COMMIT_REF e691a6dc0358c1091d59022af06a97d68fcc074d
#define QUICKCPPLIB_PREVIOUS_COMMIT_DATE "2021-09-15 10:28:22 +00:00"
#define QUICKCPPLIB_PREVIOUS_COMMIT_UNIQUE e691a6dc
#endif
#define QUICKCPPLIB_VERSION_GLUE2(a, b) a##b
#define QUICKCPPLIB_VERSION_GLUE(a, b) QUICKCPPLIB_VERSION_GLUE2(a, b)
// clang-format off
#if defined(QUICKCPPLIB_DISABLE_ABI_PERMUTATION)
#define QUICKCPPLIB_NAMESPACE quickcpplib
#define QUICKCPPLIB_NAMESPACE_BEGIN namespace quickcpplib {
#define QUICKCPPLIB_NAMESPACE_END }
#else
#define QUICKCPPLIB_NAMESPACE quickcpplib::QUICKCPPLIB_VERSION_GLUE(_, QUICKCPPLIB_PREVIOUS_COMMIT_UNIQUE)
#define QUICKCPPLIB_NAMESPACE_BEGIN namespace quickcpplib { namespace QUICKCPPLIB_VERSION_GLUE(_, QUICKCPPLIB_PREVIOUS_COMMIT_UNIQUE) {
#define QUICKCPPLIB_NAMESPACE_END } }
#endif
// clang-format on
#ifdef _MSC_VER
#define QUICKCPPLIB_BIND_MESSAGE_PRAGMA2(x) __pragma(message(x))
#define QUICKCPPLIB_BIND_MESSAGE_PRAGMA(x) QUICKCPPLIB_BIND_MESSAGE_PRAGMA2(x)
#define QUICKCPPLIB_BIND_MESSAGE_PREFIX(type) __FILE__ "(" QUICKCPPLIB_BIND_STRINGIZE2(__LINE__) "): " type ": "
#define QUICKCPPLIB_BIND_MESSAGE_(type, prefix, msg) QUICKCPPLIB_BIND_MESSAGE_PRAGMA(prefix msg)
#else
#define QUICKCPPLIB_BIND_MESSAGE_PRAGMA2(x) _Pragma(#x)
#define QUICKCPPLIB_BIND_MESSAGE_PRAGMA(type, x) QUICKCPPLIB_BIND_MESSAGE_PRAGMA2(type x)
#define QUICKCPPLIB_BIND_MESSAGE_(type, prefix, msg) QUICKCPPLIB_BIND_MESSAGE_PRAGMA(type, msg)
#endif
//! Have the compiler output a message
#define QUICKCPPLIB_MESSAGE(msg) QUICKCPPLIB_BIND_MESSAGE_(message, QUICKCPPLIB_BIND_MESSAGE_PREFIX("message"), msg)
//! Have the compiler output a note
#define QUICKCPPLIB_NOTE(msg) QUICKCPPLIB_BIND_MESSAGE_(message, QUICKCPPLIB_BIND_MESSAGE_PREFIX("note"), msg)
//! Have the compiler output a warning
#define QUICKCPPLIB_WARNING(msg) QUICKCPPLIB_BIND_MESSAGE_(GCC warning, QUICKCPPLIB_BIND_MESSAGE_PREFIX("warning"), msg)
//! Have the compiler output an error
#define QUICKCPPLIB_ERROR(msg) QUICKCPPLIB_BIND_MESSAGE_(GCC error, QUICKCPPLIB_BIND_MESSAGE_PREFIX("error"), msg)
#define QUICKCPPLIB_ANNOTATE_RWLOCK_CREATE(p)
#define QUICKCPPLIB_ANNOTATE_RWLOCK_DESTROY(p)
#define QUICKCPPLIB_ANNOTATE_RWLOCK_ACQUIRED(p, s)
#define QUICKCPPLIB_ANNOTATE_RWLOCK_RELEASED(p, s)
#define QUICKCPPLIB_ANNOTATE_IGNORE_READS_BEGIN()
#define QUICKCPPLIB_ANNOTATE_IGNORE_READS_END()
#define QUICKCPPLIB_ANNOTATE_IGNORE_WRITES_BEGIN()
#define QUICKCPPLIB_ANNOTATE_IGNORE_WRITES_END()
#define QUICKCPPLIB_DRD_IGNORE_VAR(x)
#define QUICKCPPLIB_DRD_STOP_IGNORING_VAR(x)
#define QUICKCPPLIB_RUNNING_ON_VALGRIND (0)
#ifndef QUICKCPPLIB_IN_THREAD_SANITIZER
#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define QUICKCPPLIB_IN_THREAD_SANITIZER 1
#endif
#elif defined(__SANITIZE_THREAD__)
#define QUICKCPPLIB_IN_THREAD_SANITIZER 1
#endif
#endif
#ifndef QUICKCPPLIB_IN_THREAD_SANITIZER
#define QUICKCPPLIB_IN_THREAD_SANITIZER 0
#endif
#if QUICKCPPLIB_IN_THREAD_SANITIZER
#define QUICKCPPLIB_DISABLE_THREAD_SANITIZE __attribute__((no_sanitize_thread))
#else
#define QUICKCPPLIB_DISABLE_THREAD_SANITIZE
#endif
#ifndef QUICKCPPLIB_SMT_PAUSE
#if !defined(__clang__) && defined(_MSC_VER) && _MSC_VER >= 1310 && (defined(_M_IX86) || defined(_M_X64))
extern "C" void _mm_pause();
#pragma intrinsic(_mm_pause)
#define QUICKCPPLIB_SMT_PAUSE _mm_pause();
#elif !defined(__c2__) && defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))
#define QUICKCPPLIB_SMT_PAUSE __asm__ __volatile__("rep; nop" : : : "memory");
#endif
#endif
#ifndef QUICKCPPLIB_FORCEINLINE
#if defined(_MSC_VER)
#define QUICKCPPLIB_FORCEINLINE __forceinline
#elif defined(__GNUC__)
#define QUICKCPPLIB_FORCEINLINE __attribute__((always_inline))
#else
#define QUICKCPPLIB_FORCEINLINE
#endif
#endif
#ifndef QUICKCPPLIB_NOINLINE
#if defined(_MSC_VER)
#define QUICKCPPLIB_NOINLINE __declspec(noinline)
#elif defined(__GNUC__)
#define QUICKCPPLIB_NOINLINE __attribute__((noinline))
#else
#define QUICKCPPLIB_NOINLINE
#endif
#endif
#ifdef __has_cpp_attribute
#define QUICKCPPLIB_HAS_CPP_ATTRIBUTE(attr) __has_cpp_attribute(attr)
#else
#define QUICKCPPLIB_HAS_CPP_ATTRIBUTE(attr) (0)
#endif
#if !defined(QUICKCPPLIB_NORETURN)
#if QUICKCPPLIB_HAS_CPP_ATTRIBUTE(noreturn)
#define QUICKCPPLIB_NORETURN [[noreturn]]
#elif defined(_MSC_VER)
#define QUICKCPPLIB_NORETURN __declspec(noreturn)
#elif defined(__GNUC__)
#define QUICKCPPLIB_NORETURN __attribute__((__noreturn__))
#else
#define QUICKCPPLIB_NORETURN
#endif
#endif
#ifndef QUICKCPPLIB_NODISCARD
#if 0 || (_HAS_CXX17 && _MSC_VER >= 1911 /* VS2017.3 */)
#define QUICKCPPLIB_NODISCARD [[nodiscard]]
#endif
#endif
#ifndef QUICKCPPLIB_NODISCARD
#if QUICKCPPLIB_HAS_CPP_ATTRIBUTE(nodiscard)
#define QUICKCPPLIB_NODISCARD [[nodiscard]]
#elif defined(__clang__) // deliberately not GCC
#define QUICKCPPLIB_NODISCARD __attribute__((warn_unused_result))
#elif defined(_MSC_VER)
// _Must_inspect_result_ expands into this
#define QUICKCPPLIB_NODISCARD __declspec("SAL_name" "(" "\"_Must_inspect_result_\"" "," "\"\"" "," "\"2\"" ")") __declspec("SAL_begin") __declspec("SAL_post") __declspec("SAL_mustInspect") __declspec("SAL_post") __declspec("SAL_checkReturn") __declspec("SAL_end")
#endif
#endif
#ifndef QUICKCPPLIB_NODISCARD
#define QUICKCPPLIB_NODISCARD
#endif
#ifndef QUICKCPPLIB_SYMBOL_VISIBLE
#if defined(_MSC_VER)
#define QUICKCPPLIB_SYMBOL_VISIBLE
#elif defined(__GNUC__)
#define QUICKCPPLIB_SYMBOL_VISIBLE __attribute__((visibility("default")))
#else
#define QUICKCPPLIB_SYMBOL_VISIBLE
#endif
#endif
#ifndef QUICKCPPLIB_SYMBOL_EXPORT
#if defined(_MSC_VER)
#define QUICKCPPLIB_SYMBOL_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define QUICKCPPLIB_SYMBOL_EXPORT __attribute__((visibility("default")))
#else
#define QUICKCPPLIB_SYMBOL_EXPORT
#endif
#endif
#ifndef QUICKCPPLIB_SYMBOL_IMPORT
#if defined(_MSC_VER)
#define QUICKCPPLIB_SYMBOL_IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
#define QUICKCPPLIB_SYMBOL_IMPORT
#else
#define QUICKCPPLIB_SYMBOL_IMPORT
#endif
#endif
#ifndef QUICKCPPLIB_THREAD_LOCAL
#if _MSC_VER >= 1800
#define QUICKCPPLIB_THREAD_LOCAL_IS_CXX11 1
#elif __cplusplus >= 201103
#if __GNUC__ >= 5 && !defined(__clang__)
#define QUICKCPPLIB_THREAD_LOCAL_IS_CXX11 1
#elif defined(__has_feature)
#if __has_feature(cxx_thread_local)
#define QUICKCPPLIB_THREAD_LOCAL_IS_CXX11 1
#endif
#endif
#endif
#ifdef QUICKCPPLIB_THREAD_LOCAL_IS_CXX11
#define QUICKCPPLIB_THREAD_LOCAL thread_local
#endif
#ifndef QUICKCPPLIB_THREAD_LOCAL
#if defined(_MSC_VER)
#define QUICKCPPLIB_THREAD_LOCAL __declspec(thread)
#elif defined(__GNUC__)
#define QUICKCPPLIB_THREAD_LOCAL __thread
#else
#error Unknown compiler, cannot set QUICKCPPLIB_THREAD_LOCAL
#endif
#endif
#endif
/* MSVC capable preprocessor macro overloading
(C) 2014-2017 Niall Douglas <http://www.nedproductions.biz/> (3 commits)
File Created: Aug 2014


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef QUICKCPPLIB_PREPROCESSOR_MACRO_OVERLOAD_H
#define QUICKCPPLIB_PREPROCESSOR_MACRO_OVERLOAD_H
#define QUICKCPPLIB_GLUE(x, y) x y
#define QUICKCPPLIB_RETURN_ARG_COUNT(_1_, _2_, _3_, _4_, _5_, _6_, _7_, _8_, count, ...) count
#define QUICKCPPLIB_EXPAND_ARGS(args) QUICKCPPLIB_RETURN_ARG_COUNT args
#define QUICKCPPLIB_COUNT_ARGS_MAX8(...) QUICKCPPLIB_EXPAND_ARGS((__VA_ARGS__, 8, 7, 6, 5, 4, 3, 2, 1, 0))
#define QUICKCPPLIB_OVERLOAD_MACRO2(name, count) name##count
#define QUICKCPPLIB_OVERLOAD_MACRO1(name, count) QUICKCPPLIB_OVERLOAD_MACRO2(name, count)
#define QUICKCPPLIB_OVERLOAD_MACRO(name, count) QUICKCPPLIB_OVERLOAD_MACRO1(name, count)
#define QUICKCPPLIB_CALL_OVERLOAD(name, ...) QUICKCPPLIB_GLUE(QUICKCPPLIB_OVERLOAD_MACRO(name, QUICKCPPLIB_COUNT_ARGS_MAX8(__VA_ARGS__)), (__VA_ARGS__))
#define QUICKCPPLIB_GLUE_(x, y) x y
#define QUICKCPPLIB_RETURN_ARG_COUNT_(_1_, _2_, _3_, _4_, _5_, _6_, _7_, _8_, count, ...) count
#define QUICKCPPLIB_EXPAND_ARGS_(args) QUICKCPPLIB_RETURN_ARG_COUNT_ args
#define QUICKCPPLIB_COUNT_ARGS_MAX8_(...) QUICKCPPLIB_EXPAND_ARGS_((__VA_ARGS__, 8, 7, 6, 5, 4, 3, 2, 1, 0))
#define QUICKCPPLIB_OVERLOAD_MACRO2_(name, count) name##count
#define QUICKCPPLIB_OVERLOAD_MACRO1_(name, count) QUICKCPPLIB_OVERLOAD_MACRO2_(name, count)
#define QUICKCPPLIB_OVERLOAD_MACRO_(name, count) QUICKCPPLIB_OVERLOAD_MACRO1_(name, count)
#define QUICKCPPLIB_CALL_OVERLOAD_(name, ...) QUICKCPPLIB_GLUE_(QUICKCPPLIB_OVERLOAD_MACRO_(name, QUICKCPPLIB_COUNT_ARGS_MAX8_(__VA_ARGS__)), (__VA_ARGS__))
#endif
#if defined(__cpp_concepts) && !defined(QUICKCPPLIB_DISABLE_CONCEPTS_SUPPORT)
#define QUICKCPPLIB_TREQUIRES_EXPAND8(a, b, c, d, e, f, g, h) a &&QUICKCPPLIB_TREQUIRES_EXPAND7(b, c, d, e, f, g, h)
#define QUICKCPPLIB_TREQUIRES_EXPAND7(a, b, c, d, e, f, g) a &&QUICKCPPLIB_TREQUIRES_EXPAND6(b, c, d, e, f, g)
#define QUICKCPPLIB_TREQUIRES_EXPAND6(a, b, c, d, e, f) a &&QUICKCPPLIB_TREQUIRES_EXPAND5(b, c, d, e, f)
#define QUICKCPPLIB_TREQUIRES_EXPAND5(a, b, c, d, e) a &&QUICKCPPLIB_TREQUIRES_EXPAND4(b, c, d, e)
#define QUICKCPPLIB_TREQUIRES_EXPAND4(a, b, c, d) a &&QUICKCPPLIB_TREQUIRES_EXPAND3(b, c, d)
#define QUICKCPPLIB_TREQUIRES_EXPAND3(a, b, c) a &&QUICKCPPLIB_TREQUIRES_EXPAND2(b, c)
#define QUICKCPPLIB_TREQUIRES_EXPAND2(a, b) a &&QUICKCPPLIB_TREQUIRES_EXPAND1(b)
#define QUICKCPPLIB_TREQUIRES_EXPAND1(a) a
//! Expands into a && b && c && ...
#define QUICKCPPLIB_TREQUIRES(...) requires QUICKCPPLIB_CALL_OVERLOAD(QUICKCPPLIB_TREQUIRES_EXPAND, __VA_ARGS__)
#define QUICKCPPLIB_TEMPLATE(...) template <__VA_ARGS__>
#define QUICKCPPLIB_TEXPR(...) requires { (__VA_ARGS__); }
#define QUICKCPPLIB_TPRED(...) (__VA_ARGS__)
#if !defined(_MSC_VER) || _MSC_FULL_VER >= 192400000 // VS 2019 16.3 is broken here
#define QUICKCPPLIB_REQUIRES(...) requires(__VA_ARGS__)
#else
#define QUICKCPPLIB_REQUIRES(...)
#endif
#else
#define QUICKCPPLIB_TEMPLATE(...) template <__VA_ARGS__
#define QUICKCPPLIB_TREQUIRES(...) , __VA_ARGS__ >
#define QUICKCPPLIB_TEXPR(...) typename = decltype(__VA_ARGS__)
#ifdef _MSC_VER
// MSVC gives an error if every specialisation of a template is always ill-formed, so
// the more powerful SFINAE form below causes pukeage :(
#define QUICKCPPLIB_TPRED(...) typename = typename std::enable_if<(__VA_ARGS__)>::type
#else
#define QUICKCPPLIB_TPRED(...) typename std::enable_if<(__VA_ARGS__), bool>::type = true
#endif
#define QUICKCPPLIB_REQUIRES(...)
#endif
#endif
#ifndef __cpp_variadic_templates
#error Outcome needs variadic template support in the compiler
#endif
#if __cpp_constexpr < 201304 && _MSC_FULL_VER < 191100000
#error Outcome needs constexpr (C++ 14) support in the compiler
#endif
#ifndef __cpp_variable_templates
#error Outcome needs variable template support in the compiler
#endif
#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ < 6
#error Due to a bug in nested template variables parsing, Outcome does not work on GCCs earlier than v6.
#endif
#ifndef OUTCOME_SYMBOL_VISIBLE
#define OUTCOME_SYMBOL_VISIBLE QUICKCPPLIB_SYMBOL_VISIBLE
#endif
#ifndef OUTCOME_FORCEINLINE
#define OUTCOME_FORCEINLINE QUICKCPPLIB_FORCEINLINE
#endif
#ifndef OUTCOME_NODISCARD
#define OUTCOME_NODISCARD QUICKCPPLIB_NODISCARD
#endif
#ifndef OUTCOME_THREAD_LOCAL
#define OUTCOME_THREAD_LOCAL QUICKCPPLIB_THREAD_LOCAL
#endif
#ifndef OUTCOME_TEMPLATE
#define OUTCOME_TEMPLATE(...) QUICKCPPLIB_TEMPLATE(__VA_ARGS__)
#endif
#ifndef OUTCOME_TREQUIRES
#define OUTCOME_TREQUIRES(...) QUICKCPPLIB_TREQUIRES(__VA_ARGS__)
#endif
#ifndef OUTCOME_TEXPR
#define OUTCOME_TEXPR(...) QUICKCPPLIB_TEXPR(__VA_ARGS__)
#endif
#ifndef OUTCOME_TPRED
#define OUTCOME_TPRED(...) QUICKCPPLIB_TPRED(__VA_ARGS__)
#endif
#ifndef OUTCOME_REQUIRES
#define OUTCOME_REQUIRES(...) QUICKCPPLIB_REQUIRES(__VA_ARGS__)
#endif
/* Convenience macros for importing local namespace binds
(C) 2014-2017 Niall Douglas <http://www.nedproductions.biz/> (9 commits)
File Created: Aug 2014


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef QUICKCPPLIB_BIND_IMPORT_HPP
#define QUICKCPPLIB_BIND_IMPORT_HPP
/* 2014-10-9 ned: I lost today figuring out the below. I really hate the C preprocessor now.
 *
 * Anyway, infinity = 8. It's easy to expand below if needed.
 */
#define QUICKCPPLIB_BIND_STRINGIZE(a) #a
#define QUICKCPPLIB_BIND_STRINGIZE2(a) QUICKCPPLIB_BIND_STRINGIZE(a)
#define QUICKCPPLIB_BIND_NAMESPACE_VERSION8(a, b, c, d, e, f, g, h) a##_##b##_##c##_##d##_##e##_##f##_##g##_##h
#define QUICKCPPLIB_BIND_NAMESPACE_VERSION7(a, b, c, d, e, f, g) a##_##b##_##c##_##d##_##e##_##f##_##g
#define QUICKCPPLIB_BIND_NAMESPACE_VERSION6(a, b, c, d, e, f) a##_##b##_##c##_##d##_##e##_##f
#define QUICKCPPLIB_BIND_NAMESPACE_VERSION5(a, b, c, d, e) a##_##b##_##c##_##d##_##e
#define QUICKCPPLIB_BIND_NAMESPACE_VERSION4(a, b, c, d) a##_##b##_##c##_##d
#define QUICKCPPLIB_BIND_NAMESPACE_VERSION3(a, b, c) a##_##b##_##c
#define QUICKCPPLIB_BIND_NAMESPACE_VERSION2(a, b) a##_##b
#define QUICKCPPLIB_BIND_NAMESPACE_VERSION1(a) a
//! Concatenates each parameter with _
#define QUICKCPPLIB_BIND_NAMESPACE_VERSION(...) QUICKCPPLIB_CALL_OVERLOAD(QUICKCPPLIB_BIND_NAMESPACE_VERSION, __VA_ARGS__)
#define QUICKCPPLIB_BIND_NAMESPACE_SELECT_2(name, modifier) name
#define QUICKCPPLIB_BIND_NAMESPACE_SELECT2(name, modifier) ::name
#define QUICKCPPLIB_BIND_NAMESPACE_SELECT_1(name) name
#define QUICKCPPLIB_BIND_NAMESPACE_SELECT1(name) ::name
#define QUICKCPPLIB_BIND_NAMESPACE_SELECT_(...) QUICKCPPLIB_CALL_OVERLOAD_(QUICKCPPLIB_BIND_NAMESPACE_SELECT_, __VA_ARGS__)
#define QUICKCPPLIB_BIND_NAMESPACE_SELECT(...) QUICKCPPLIB_CALL_OVERLOAD_(QUICKCPPLIB_BIND_NAMESPACE_SELECT, __VA_ARGS__)
#define QUICKCPPLIB_BIND_NAMESPACE_EXPAND8(a, b, c, d, e, f, g, h) QUICKCPPLIB_BIND_NAMESPACE_SELECT_ a QUICKCPPLIB_BIND_NAMESPACE_SELECT b QUICKCPPLIB_BIND_NAMESPACE_SELECT c QUICKCPPLIB_BIND_NAMESPACE_SELECT d QUICKCPPLIB_BIND_NAMESPACE_SELECT e QUICKCPPLIB_BIND_NAMESPACE_SELECT f QUICKCPPLIB_BIND_NAMESPACE_SELECT g QUICKCPPLIB_BIND_NAMESPACE_SELECT h
#define QUICKCPPLIB_BIND_NAMESPACE_EXPAND7(a, b, c, d, e, f, g) QUICKCPPLIB_BIND_NAMESPACE_SELECT_ a QUICKCPPLIB_BIND_NAMESPACE_SELECT b QUICKCPPLIB_BIND_NAMESPACE_SELECT c QUICKCPPLIB_BIND_NAMESPACE_SELECT d QUICKCPPLIB_BIND_NAMESPACE_SELECT e QUICKCPPLIB_BIND_NAMESPACE_SELECT f QUICKCPPLIB_BIND_NAMESPACE_SELECT g
#define QUICKCPPLIB_BIND_NAMESPACE_EXPAND6(a, b, c, d, e, f) QUICKCPPLIB_BIND_NAMESPACE_SELECT_ a QUICKCPPLIB_BIND_NAMESPACE_SELECT b QUICKCPPLIB_BIND_NAMESPACE_SELECT c QUICKCPPLIB_BIND_NAMESPACE_SELECT d QUICKCPPLIB_BIND_NAMESPACE_SELECT e QUICKCPPLIB_BIND_NAMESPACE_SELECT f
#define QUICKCPPLIB_BIND_NAMESPACE_EXPAND5(a, b, c, d, e) QUICKCPPLIB_BIND_NAMESPACE_SELECT_ a QUICKCPPLIB_BIND_NAMESPACE_SELECT b QUICKCPPLIB_BIND_NAMESPACE_SELECT c QUICKCPPLIB_BIND_NAMESPACE_SELECT d QUICKCPPLIB_BIND_NAMESPACE_SELECT e
#define QUICKCPPLIB_BIND_NAMESPACE_EXPAND4(a, b, c, d) QUICKCPPLIB_BIND_NAMESPACE_SELECT_ a QUICKCPPLIB_BIND_NAMESPACE_SELECT b QUICKCPPLIB_BIND_NAMESPACE_SELECT c QUICKCPPLIB_BIND_NAMESPACE_SELECT d
#define QUICKCPPLIB_BIND_NAMESPACE_EXPAND3(a, b, c) QUICKCPPLIB_BIND_NAMESPACE_SELECT_ a QUICKCPPLIB_BIND_NAMESPACE_SELECT b QUICKCPPLIB_BIND_NAMESPACE_SELECT c
#define QUICKCPPLIB_BIND_NAMESPACE_EXPAND2(a, b) QUICKCPPLIB_BIND_NAMESPACE_SELECT_ a QUICKCPPLIB_BIND_NAMESPACE_SELECT b
#define QUICKCPPLIB_BIND_NAMESPACE_EXPAND1(a) QUICKCPPLIB_BIND_NAMESPACE_SELECT_ a
//! Expands into a::b::c:: ...
#define QUICKCPPLIB_BIND_NAMESPACE(...) QUICKCPPLIB_CALL_OVERLOAD(QUICKCPPLIB_BIND_NAMESPACE_EXPAND, __VA_ARGS__)
#define QUICKCPPLIB_BIND_NAMESPACE_BEGIN_NAMESPACE_SELECT2(name, modifier) modifier namespace name {
#define QUICKCPPLIB_BIND_NAMESPACE_BEGIN_NAMESPACE_SELECT1(name) namespace name {
#define QUICKCPPLIB_BIND_NAMESPACE_BEGIN_NAMESPACE_SELECT(...) QUICKCPPLIB_CALL_OVERLOAD_(QUICKCPPLIB_BIND_NAMESPACE_BEGIN_NAMESPACE_SELECT, __VA_ARGS__)
#define QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND8(a, b, c, d, e, f, g, h) QUICKCPPLIB_BIND_NAMESPACE_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND7(b, c, d, e, f, g, h)
#define QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND7(a, b, c, d, e, f, g) QUICKCPPLIB_BIND_NAMESPACE_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND6(b, c, d, e, f, g)
#define QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND6(a, b, c, d, e, f) QUICKCPPLIB_BIND_NAMESPACE_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND5(b, c, d, e, f)
#define QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND5(a, b, c, d, e) QUICKCPPLIB_BIND_NAMESPACE_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND4(b, c, d, e)
#define QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND4(a, b, c, d) QUICKCPPLIB_BIND_NAMESPACE_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND3(b, c, d)
#define QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND3(a, b, c) QUICKCPPLIB_BIND_NAMESPACE_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND2(b, c)
#define QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND2(a, b) QUICKCPPLIB_BIND_NAMESPACE_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND1(b)
#define QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND1(a) QUICKCPPLIB_BIND_NAMESPACE_BEGIN_NAMESPACE_SELECT a
//! Expands into namespace a { namespace b { namespace c ...
#define QUICKCPPLIB_BIND_NAMESPACE_BEGIN(...) QUICKCPPLIB_CALL_OVERLOAD(QUICKCPPLIB_BIND_NAMESPACE_BEGIN_EXPAND, __VA_ARGS__)
#define QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_NAMESPACE_SELECT2(name, modifier) modifier namespace name {
#define QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_NAMESPACE_SELECT1(name) export namespace name {
#define QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_NAMESPACE_SELECT(...) QUICKCPPLIB_CALL_OVERLOAD_(QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_NAMESPACE_SELECT, __VA_ARGS__)
#define QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND8(a, b, c, d, e, f, g, h) QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND7(b, c, d, e, f, g, h)
#define QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND7(a, b, c, d, e, f, g) QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND6(b, c, d, e, f, g)
#define QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND6(a, b, c, d, e, f) QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND5(b, c, d, e, f)
#define QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND5(a, b, c, d, e) QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND4(b, c, d, e)
#define QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND4(a, b, c, d) QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND3(b, c, d)
#define QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND3(a, b, c) QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND2(b, c)
#define QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND2(a, b) QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND1(b)
#define QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND1(a) QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_NAMESPACE_SELECT a
//! Expands into export namespace a { namespace b { namespace c ...
#define QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN(...) QUICKCPPLIB_CALL_OVERLOAD(QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN_EXPAND, __VA_ARGS__)
#define QUICKCPPLIB_BIND_NAMESPACE_END_NAMESPACE_SELECT2(name, modifier) }
#define QUICKCPPLIB_BIND_NAMESPACE_END_NAMESPACE_SELECT1(name) }
#define QUICKCPPLIB_BIND_NAMESPACE_END_NAMESPACE_SELECT(...) QUICKCPPLIB_CALL_OVERLOAD_(QUICKCPPLIB_BIND_NAMESPACE_END_NAMESPACE_SELECT, __VA_ARGS__)
#define QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND8(a, b, c, d, e, f, g, h) QUICKCPPLIB_BIND_NAMESPACE_END_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND7(b, c, d, e, f, g, h)
#define QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND7(a, b, c, d, e, f, g) QUICKCPPLIB_BIND_NAMESPACE_END_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND6(b, c, d, e, f, g)
#define QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND6(a, b, c, d, e, f) QUICKCPPLIB_BIND_NAMESPACE_END_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND5(b, c, d, e, f)
#define QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND5(a, b, c, d, e) QUICKCPPLIB_BIND_NAMESPACE_END_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND4(b, c, d, e)
#define QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND4(a, b, c, d) QUICKCPPLIB_BIND_NAMESPACE_END_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND3(b, c, d)
#define QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND3(a, b, c) QUICKCPPLIB_BIND_NAMESPACE_END_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND2(b, c)
#define QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND2(a, b) QUICKCPPLIB_BIND_NAMESPACE_END_NAMESPACE_SELECT a QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND1(b)
#define QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND1(a) QUICKCPPLIB_BIND_NAMESPACE_END_NAMESPACE_SELECT a
//! Expands into } } ...
#define QUICKCPPLIB_BIND_NAMESPACE_END(...) QUICKCPPLIB_CALL_OVERLOAD(QUICKCPPLIB_BIND_NAMESPACE_END_EXPAND, __VA_ARGS__)
//! Expands into a static const char string array used to mark BindLib compatible namespaces
#define QUICKCPPLIB_BIND_DECLARE(decl, desc) static const char *quickcpplib_out[] = {#decl, desc};
#endif
#ifndef OUTCOME_ENABLE_LEGACY_SUPPORT_FOR
#define OUTCOME_ENABLE_LEGACY_SUPPORT_FOR 220 // the v2.2 Outcome release
#endif
#if defined(OUTCOME_UNSTABLE_VERSION)
/* UPDATED BY SCRIPT
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (225 commits)


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
// Note the second line of this file must ALWAYS be the git SHA, third line ALWAYS the git SHA update time
#define OUTCOME_PREVIOUS_COMMIT_REF e261cebddfd2d5d1229dbf66c6dc0091a9f2a6f8
#define OUTCOME_PREVIOUS_COMMIT_DATE "2021-10-26 10:23:56 +00:00"
#define OUTCOME_PREVIOUS_COMMIT_UNIQUE e261cebd
#define OUTCOME_V2 (QUICKCPPLIB_BIND_NAMESPACE_VERSION(outcome_v2, OUTCOME_PREVIOUS_COMMIT_UNIQUE))
#ifdef _DEBUG
#define OUTCOME_V2_CXX_MODULE_NAME QUICKCPPLIB_BIND_NAMESPACE((QUICKCPPLIB_BIND_NAMESPACE_VERSION(outcome_v2d, OUTCOME_PREVIOUS_COMMIT_UNIQUE)))
#else
#define OUTCOME_V2_CXX_MODULE_NAME QUICKCPPLIB_BIND_NAMESPACE((QUICKCPPLIB_BIND_NAMESPACE_VERSION(outcome_v2, OUTCOME_PREVIOUS_COMMIT_UNIQUE)))
#endif
#else
#define OUTCOME_V2 (QUICKCPPLIB_BIND_NAMESPACE_VERSION(outcome_v2))
#ifdef _DEBUG
#define OUTCOME_V2_CXX_MODULE_NAME QUICKCPPLIB_BIND_NAMESPACE((QUICKCPPLIB_BIND_NAMESPACE_VERSION(outcome_v2d)))
#else
#define OUTCOME_V2_CXX_MODULE_NAME QUICKCPPLIB_BIND_NAMESPACE((QUICKCPPLIB_BIND_NAMESPACE_VERSION(outcome_v2)))
#endif
#endif
#if defined(GENERATING_OUTCOME_MODULE_INTERFACE)
#define OUTCOME_V2_NAMESPACE QUICKCPPLIB_BIND_NAMESPACE(OUTCOME_V2)
#define OUTCOME_V2_NAMESPACE_BEGIN QUICKCPPLIB_BIND_NAMESPACE_BEGIN(OUTCOME_V2)
#define OUTCOME_V2_NAMESPACE_EXPORT_BEGIN QUICKCPPLIB_BIND_NAMESPACE_EXPORT_BEGIN(OUTCOME_V2)
#define OUTCOME_V2_NAMESPACE_END QUICKCPPLIB_BIND_NAMESPACE_END(OUTCOME_V2)
#else
#define OUTCOME_V2_NAMESPACE QUICKCPPLIB_BIND_NAMESPACE(OUTCOME_V2)
#define OUTCOME_V2_NAMESPACE_BEGIN QUICKCPPLIB_BIND_NAMESPACE_BEGIN(OUTCOME_V2)
#define OUTCOME_V2_NAMESPACE_EXPORT_BEGIN QUICKCPPLIB_BIND_NAMESPACE_BEGIN(OUTCOME_V2)
#define OUTCOME_V2_NAMESPACE_END QUICKCPPLIB_BIND_NAMESPACE_END(OUTCOME_V2)
#endif
#include <cstdint> // for uint32_t etc
#include <initializer_list>
#include <iosfwd> // for future serialisation
#include <new> // for placement in moves etc
#include <type_traits>
#ifndef OUTCOME_USE_STD_IN_PLACE_TYPE
#if defined(_MSC_VER) && _HAS_CXX17
#define OUTCOME_USE_STD_IN_PLACE_TYPE 1 // MSVC always has std::in_place_type
#elif __cplusplus >= 201700
// libstdc++ before GCC 6 doesn't have it, despite claiming C++ 17 support
#ifdef __has_include
#if !__has_include(<variant>)
#define OUTCOME_USE_STD_IN_PLACE_TYPE 0 // must have it if <variant> is present
#endif
#endif
#ifndef OUTCOME_USE_STD_IN_PLACE_TYPE
#define OUTCOME_USE_STD_IN_PLACE_TYPE 1
#endif
#else
#define OUTCOME_USE_STD_IN_PLACE_TYPE 0
#endif
#endif
#if OUTCOME_USE_STD_IN_PLACE_TYPE
#include <utility> // for in_place_type_t
OUTCOME_V2_NAMESPACE_BEGIN
template <class T> using in_place_type_t = std::in_place_type_t<T>;
using std::in_place_type;
OUTCOME_V2_NAMESPACE_END
#else
OUTCOME_V2_NAMESPACE_BEGIN
/*! AWAITING HUGO JSON CONVERSION TOOL
type definition template <class T> in_place_type_t. Potential doc page: `in_place_type_t<T>`
*/
template <class T> struct in_place_type_t
{
  explicit in_place_type_t() = default;
};
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class T> constexpr in_place_type_t<T> in_place_type{};
OUTCOME_V2_NAMESPACE_END
#endif
#ifndef OUTCOME_TRIVIAL_ABI
#if 0 || __clang_major__ >= 7
//! Defined to be `[[clang::trivial_abi]]` when on a new enough clang compiler. Usually automatic, can be overriden.
#define OUTCOME_TRIVIAL_ABI [[clang::trivial_abi]]
#else
#define OUTCOME_TRIVIAL_ABI
#endif
#endif
OUTCOME_V2_NAMESPACE_BEGIN
namespace detail
{
  // Test if type is an in_place_type_t
  template <class T> struct is_in_place_type_t
  {
    static constexpr bool value = false;
  };
  template <class U> struct is_in_place_type_t<in_place_type_t<U>>
  {
    static constexpr bool value = true;
  };
  // Replace void with constructible void_type
  struct empty_type
  {
  };
  struct void_type
  {
    // We always compare true to another instance of me
    constexpr bool operator==(void_type /*unused*/) const noexcept { return true; }
    constexpr bool operator!=(void_type /*unused*/) const noexcept { return false; }
  };
  template <class T> using devoid = std::conditional_t<std::is_void<T>::value, void_type, T>;
  template <class Output, class Input> using rebind_type5 = Output;
  template <class Output, class Input>
  using rebind_type4 = std::conditional_t< //
  std::is_volatile<Input>::value, //
  std::add_volatile_t<rebind_type5<Output, std::remove_volatile_t<Input>>>, //
  rebind_type5<Output, Input>>;
  template <class Output, class Input>
  using rebind_type3 = std::conditional_t< //
  std::is_const<Input>::value, //
  std::add_const_t<rebind_type4<Output, std::remove_const_t<Input>>>, //
  rebind_type4<Output, Input>>;
  template <class Output, class Input>
  using rebind_type2 = std::conditional_t< //
  std::is_lvalue_reference<Input>::value, //
  std::add_lvalue_reference_t<rebind_type3<Output, std::remove_reference_t<Input>>>, //
  rebind_type3<Output, Input>>;
  template <class Output, class Input>
  using rebind_type = std::conditional_t< //
  std::is_rvalue_reference<Input>::value, //
  std::add_rvalue_reference_t<rebind_type2<Output, std::remove_reference_t<Input>>>, //
  rebind_type2<Output, Input>>;
  // static_assert(std::is_same_v<rebind_type<int, volatile const double &&>, volatile const int &&>, "");
  /* True if type is the same or constructible. Works around a bug where clang + libstdc++
  pukes on std::is_constructible<filesystem::path, void> (this bug is fixed upstream).
  */
  template <class T, class U> struct _is_explicitly_constructible
  {
    static constexpr bool value = std::is_constructible<T, U>::value;
  };
  template <class T> struct _is_explicitly_constructible<T, void>
  {
    static constexpr bool value = false;
  };
  template <> struct _is_explicitly_constructible<void, void>
  {
    static constexpr bool value = false;
  };
  template <class T, class U> static constexpr bool is_explicitly_constructible = _is_explicitly_constructible<T, U>::value;
  template <class T, class U> struct _is_implicitly_constructible
  {
    static constexpr bool value = std::is_convertible<U, T>::value;
  };
  template <class T> struct _is_implicitly_constructible<T, void>
  {
    static constexpr bool value = false;
  };
  template <> struct _is_implicitly_constructible<void, void>
  {
    static constexpr bool value = false;
  };
  template <class T, class U> static constexpr bool is_implicitly_constructible = _is_implicitly_constructible<T, U>::value;
  template <class T, class... Args> struct _is_nothrow_constructible
  {
    static constexpr bool value = std::is_nothrow_constructible<T, Args...>::value;
  };
  template <class T> struct _is_nothrow_constructible<T, void>
  {
    static constexpr bool value = false;
  };
  template <> struct _is_nothrow_constructible<void, void>
  {
    static constexpr bool value = false;
  };
  template <class T, class... Args> static constexpr bool is_nothrow_constructible = _is_nothrow_constructible<T, Args...>::value;
  template <class T, class... Args> struct _is_constructible
  {
    static constexpr bool value = std::is_constructible<T, Args...>::value;
  };
  template <class T> struct _is_constructible<T, void>
  {
    static constexpr bool value = false;
  };
  template <> struct _is_constructible<void, void>
  {
    static constexpr bool value = false;
  };
  template <class T, class... Args> static constexpr bool is_constructible = _is_constructible<T, Args...>::value;
#ifndef OUTCOME_USE_STD_IS_NOTHROW_SWAPPABLE
#if defined(_MSC_VER) && _HAS_CXX17
#define OUTCOME_USE_STD_IS_NOTHROW_SWAPPABLE 1 // MSVC always has std::is_nothrow_swappable
#elif __cplusplus >= 201700
// libstdc++ before GCC 6 doesn't have it, despite claiming C++ 17 support
#ifdef __has_include
#if !__has_include(<variant>)
#define OUTCOME_USE_STD_IS_NOTHROW_SWAPPABLE 0
#endif
#endif
#ifndef OUTCOME_USE_STD_IS_NOTHROW_SWAPPABLE
#define OUTCOME_USE_STD_IS_NOTHROW_SWAPPABLE 1
#endif
#else
#define OUTCOME_USE_STD_IS_NOTHROW_SWAPPABLE 0
#endif
#endif
// True if type is nothrow swappable
#if !0 && OUTCOME_USE_STD_IS_NOTHROW_SWAPPABLE
  template <class T> using is_nothrow_swappable = std::is_nothrow_swappable<T>;
#else
  template <class T> struct is_nothrow_swappable
  {
    static constexpr bool value = std::is_nothrow_move_constructible<T>::value && std::is_nothrow_move_assignable<T>::value;
  };
#endif
} // namespace detail
OUTCOME_V2_NAMESPACE_END
#ifndef OUTCOME_THROW_EXCEPTION
#ifdef __cpp_exceptions
#define OUTCOME_THROW_EXCEPTION(expr) throw expr
#else
#ifdef __ANDROID__
#define OUTCOME_DISABLE_EXECINFO
#endif
#ifndef OUTCOME_DISABLE_EXECINFO
#ifdef _WIN32
/* Implements backtrace() et al from glibc on win64
(C) 2016-2017 Niall Douglas <http://www.nedproductions.biz/> (4 commits)
File Created: Mar 2016


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef BOOST_BINDLIB_EXECINFO_WIN64_H
#define BOOST_BINDLIB_EXECINFO_WIN64_H
#ifndef _WIN32
#error Can only be included on Windows
#endif
#include <sal.h>
#include <stddef.h>
#ifdef QUICKCPPLIB_EXPORTS
#define EXECINFO_DECL extern __declspec(dllexport)
#else
#if defined(__cplusplus) && (!defined(QUICKCPPLIB_HEADERS_ONLY) || QUICKCPPLIB_HEADERS_ONLY == 1) && !0
#define EXECINFO_DECL inline
#elif defined(QUICKCPPLIB_DYN_LINK) && !defined(QUICKCPPLIB_STATIC_LINK)
#define EXECINFO_DECL extern __declspec(dllimport)
#else
#define EXECINFO_DECL extern
#endif
#endif
#ifdef __cplusplus
extern "C" {
#endif
//! Fill the array of void * at bt with up to len entries, returning entries filled.
EXECINFO_DECL _Check_return_ size_t backtrace(_Out_writes_(len) void **bt, _In_ size_t len);
//! Returns a malloced block of string representations of the input backtrace.
EXECINFO_DECL _Check_return_ _Ret_writes_maybenull_(len) char **backtrace_symbols(_In_reads_(len) void *const *bt, _In_ size_t len);
// extern void backtrace_symbols_fd(void *const *bt, size_t len, int fd);
#ifdef __cplusplus
}
#if (!defined(QUICKCPPLIB_HEADERS_ONLY) || QUICKCPPLIB_HEADERS_ONLY == 1) && !0
#define QUICKCPPLIB_INCLUDED_BY_HEADER 1
/* Implements backtrace() et al from glibc on win64
(C) 2016-2017 Niall Douglas <http://www.nedproductions.biz/> (14 commits)
File Created: Mar 2016


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
/* Implements backtrace() et al from glibc on win64
(C) 2016-2017 Niall Douglas <http://www.nedproductions.biz/> (4 commits)
File Created: Mar 2016


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#include <atomic>
#include <stdlib.h> // for abort
#include <string.h>
// To avoid including windows.h, this source has been macro expanded and win32 function shimmed for C++ only
#if defined(__cplusplus) && !defined(__clang__)
namespace win32
{
  extern _Ret_maybenull_ void *__stdcall LoadLibraryA(_In_ const char *lpLibFileName);
  typedef int(__stdcall *GetProcAddress_returntype)();
  extern GetProcAddress_returntype __stdcall GetProcAddress(_In_ void *hModule, _In_ const char *lpProcName);
  extern _Success_(return != 0) unsigned short __stdcall RtlCaptureStackBackTrace(_In_ unsigned long FramesToSkip, _In_ unsigned long FramesToCapture,
                                                                                  _Out_writes_to_(FramesToCapture, return ) void **BackTrace,
                                                                                  _Out_opt_ unsigned long *BackTraceHash);
  extern _Success_(return != 0)
  _When_((cchWideChar == -1) && (cbMultiByte != 0),
         _Post_equal_to_(_String_length_(lpMultiByteStr) +
                         1)) int __stdcall WideCharToMultiByte(_In_ unsigned int CodePage, _In_ unsigned long dwFlags, const wchar_t *lpWideCharStr,
                                                               _In_ int cchWideChar, _Out_writes_bytes_to_opt_(cbMultiByte, return ) char *lpMultiByteStr,
                                                               _In_ int cbMultiByte, _In_opt_ const char *lpDefaultChar, _Out_opt_ int *lpUsedDefaultChar);
#pragma comment(lib, "kernel32.lib")
#if (defined(__x86_64__) || defined(_M_X64)) || (defined(__aarch64__) || defined(_M_ARM64))
#pragma comment(linker, "/alternatename:?LoadLibraryA@win32@@YAPEAXPEBD@Z=LoadLibraryA")
#pragma comment(linker, "/alternatename:?GetProcAddress@win32@@YAP6AHXZPEAXPEBD@Z=GetProcAddress")
#pragma comment(linker, "/alternatename:?RtlCaptureStackBackTrace@win32@@YAGKKPEAPEAXPEAK@Z=RtlCaptureStackBackTrace")
#pragma comment(linker, "/alternatename:?WideCharToMultiByte@win32@@YAHIKPEB_WHPEADHPEBDPEAH@Z=WideCharToMultiByte")
#elif defined(__x86__) || defined(_M_IX86) || defined(__i386__)
#pragma comment(linker, "/alternatename:?LoadLibraryA@win32@@YGPAXPBD@Z=__imp__LoadLibraryA@4")
#pragma comment(linker, "/alternatename:?GetProcAddress@win32@@YGP6GHXZPAXPBD@Z=__imp__GetProcAddress@8")
#pragma comment(linker, "/alternatename:?RtlCaptureStackBackTrace@win32@@YGGKKPAPAXPAK@Z=__imp__RtlCaptureStackBackTrace@16")
#pragma comment(linker, "/alternatename:?WideCharToMultiByte@win32@@YGHIKPB_WHPADHPBDPAH@Z=__imp__WideCharToMultiByte@32")
#elif defined(__arm__) || defined(_M_ARM)
#pragma comment(linker, "/alternatename:?LoadLibraryA@win32@@YAPAXPBD@Z=LoadLibraryA")
#pragma comment(linker, "/alternatename:?GetProcAddress@win32@@YAP6AHXZPAXPBD@Z=GetProcAddress")
#pragma comment(linker, "/alternatename:?RtlCaptureStackBackTrace@win32@@YAGKKPAPAXPAK@Z=RtlCaptureStackBackTrace")
#pragma comment(linker, "/alternatename:?WideCharToMultiByte@win32@@YAHIKPB_WHPADHPBDPAH@Z=WideCharToMultiByte")
#else
#error Unknown architecture
#endif
} // namespace win32
#else
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif
#ifdef __cplusplus
namespace
{
#endif
  typedef struct _IMAGEHLP_LINE64
  {
    unsigned long SizeOfStruct;
    void *Key;
    unsigned long LineNumber;
    wchar_t *FileName;
    unsigned long long int Address;
  } IMAGEHLP_LINE64, *PIMAGEHLP_LINE64;
  typedef int(__stdcall *SymInitialize_t)(_In_ void *hProcess, _In_opt_ const wchar_t *UserSearchPath, _In_ int fInvadeProcess);
  typedef int(__stdcall *SymGetLineFromAddr64_t)(_In_ void *hProcess, _In_ unsigned long long int dwAddr, _Out_ unsigned long *pdwDisplacement,
                                                 _Out_ PIMAGEHLP_LINE64 Line);
  static std::atomic<unsigned> dbghelp_init_lock;
#if defined(__cplusplus) && !defined(__clang__)
  static void *dbghelp;
#else
static HMODULE dbghelp;
#endif
  static SymInitialize_t SymInitialize;
  static SymGetLineFromAddr64_t SymGetLineFromAddr64;
  static void load_dbghelp()
  {
#if defined(__cplusplus) && !defined(__clang__)
    using win32::GetProcAddress;
    using win32::LoadLibraryA;
#endif
    while(dbghelp_init_lock.exchange(1, std::memory_order_acq_rel))
      ;
    if(dbghelp)
    {
      dbghelp_init_lock.store(0, std::memory_order_release);
      return;
    }
    dbghelp = LoadLibraryA("DBGHELP.DLL");
    if(dbghelp)
    {
      SymInitialize = (SymInitialize_t) GetProcAddress(dbghelp, "SymInitializeW");
      if(!SymInitialize)
        abort();
      if(!SymInitialize((void *) (size_t) -1 /*GetCurrentProcess()*/, NULL, 1))
        abort();
      SymGetLineFromAddr64 = (SymGetLineFromAddr64_t) GetProcAddress(dbghelp, "SymGetLineFromAddrW64");
      if(!SymGetLineFromAddr64)
        abort();
    }
    dbghelp_init_lock.store(0, std::memory_order_release);
  }
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C"
{
#endif
  _Check_return_ size_t backtrace(_Out_writes_(len) void **bt, _In_ size_t len)
  {
#if defined(__cplusplus) && !defined(__clang__)
    using win32::RtlCaptureStackBackTrace;
#endif
    return RtlCaptureStackBackTrace(1, (unsigned long) len, bt, NULL);
  }
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 6385 6386) // MSVC static analyser can't grok this function. clang's analyser gives it thumbs up.
#endif
  _Check_return_ _Ret_writes_maybenull_(len) char **backtrace_symbols(_In_reads_(len) void *const *bt, _In_ size_t len)
  {
#if defined(__cplusplus) && !defined(__clang__)
    using win32::WideCharToMultiByte;
#endif
    size_t bytes = (len + 1) * sizeof(void *) + 256, n;
    if(!len)
      return NULL;
    else
    {
      char **ret = (char **) malloc(bytes);
      char *p = (char *) (ret + len + 1), *end = (char *) ret + bytes;
      if(!ret)
        return NULL;
      for(n = 0; n < len + 1; n++)
        ret[n] = NULL;
      load_dbghelp();
      for(n = 0; n < len; n++)
      {
        unsigned long displ;
        IMAGEHLP_LINE64 ihl;
        memset(&ihl, 0, sizeof(ihl));
        ihl.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
        int please_realloc = 0;
        if(!bt[n])
        {
          ret[n] = NULL;
        }
        else
        {
          // Keep offset till later
          ret[n] = (char *) ((char *) p - (char *) ret);
          {
            static std::atomic<unsigned> symlock(0);
            while(symlock.exchange(1, std::memory_order_acq_rel))
              ;
            if(!SymGetLineFromAddr64 || !SymGetLineFromAddr64((void *) (size_t) -1 /*GetCurrentProcess()*/, (size_t) bt[n], &displ, &ihl))
            {
              symlock.store(0, std::memory_order_release);
              if(n == 0)
              {
                free(ret);
                return NULL;
              }
              ihl.FileName = (wchar_t *) L"unknown";
              ihl.LineNumber = 0;
            }
            else
            {
              symlock.store(0, std::memory_order_release);
            }
          }
        retry:
          if(please_realloc)
          {
            char **temp = (char **) realloc(ret, bytes + 256);
            if(!temp)
            {
              free(ret);
              return NULL;
            }
            p = (char *) temp + (p - (char *) ret);
            ret = temp;
            bytes += 256;
            end = (char *) ret + bytes;
          }
          if(ihl.FileName && ihl.FileName[0])
          {
            int plen = WideCharToMultiByte(65001 /*CP_UTF8*/, 0, ihl.FileName, -1, p, (int) (end - p), NULL, NULL);
            if(!plen)
            {
              please_realloc = 1;
              goto retry;
            }
            p[plen - 1] = 0;
            p += plen - 1;
          }
          else
          {
            if(end - p < 16)
            {
              please_realloc = 1;
              goto retry;
            }
            _ui64toa_s((size_t) bt[n], p, end - p, 16);
            p = strchr(p, 0);
          }
          if(end - p < 16)
          {
            please_realloc = 1;
            goto retry;
          }
          *p++ = ':';
          _itoa_s(ihl.LineNumber, p, end - p, 10);
          p = strchr(p, 0) + 1;
        }
      }
      for(n = 0; n < len; n++)
      {
        if(ret[n])
          ret[n] = (char *) ret + (size_t) ret[n];
      }
      return ret;
    }
  }
#ifdef _MSC_VER
#pragma warning(pop)
#endif
  // extern void backtrace_symbols_fd(void *const *bt, size_t len, int fd);
#ifdef __cplusplus
}
#endif
#undef QUICKCPPLIB_INCLUDED_BY_HEADER
#endif
#endif
#endif
#else
#include <execinfo.h>
#endif
#endif // OUTCOME_DISABLE_EXECINFO
#include <cstdio>
#include <cstdlib>
OUTCOME_V2_NAMESPACE_BEGIN
namespace detail
{
  QUICKCPPLIB_NORETURN inline void do_fatal_exit(const char *expr)
  {
#if !defined(OUTCOME_DISABLE_EXECINFO)
    void *bt[16];
    size_t btlen = backtrace(bt, sizeof(bt) / sizeof(bt[0])); // NOLINT
#endif
    fprintf(stderr, "FATAL: Outcome throws exception %s with exceptions disabled\n", expr); // NOLINT
#if !defined(OUTCOME_DISABLE_EXECINFO)
    char **bts = backtrace_symbols(bt, btlen); // NOLINT
    if(bts != nullptr)
    {
      for(size_t n = 0; n < btlen; n++)
      {
        fprintf(stderr, "  %s\n", bts[n]); // NOLINT
      }
      free(bts); // NOLINT
    }
#endif
    abort();
  }
} // namespace detail
OUTCOME_V2_NAMESPACE_END
#define OUTCOME_THROW_EXCEPTION(expr) OUTCOME_V2_NAMESPACE::detail::do_fatal_exit(#expr), (void) (expr)
#endif
#endif
#ifndef BOOST_OUTCOME_AUTO_TEST_CASE
#define BOOST_OUTCOME_AUTO_TEST_CASE(a, b) BOOST_AUTO_TEST_CASE(a, b)
#endif
#endif
#define OUTCOME_COROUTINE_SUPPORT_NAMESPACE_BEGIN OUTCOME_V2_NAMESPACE_BEGIN namespace awaitables {
//
#define OUTCOME_COROUTINE_SUPPORT_NAMESPACE_EXPORT_BEGIN OUTCOME_V2_NAMESPACE_EXPORT_BEGIN namespace awaitables {
//
#define OUTCOME_COROUTINE_SUPPORT_NAMESPACE_END } OUTCOME_V2_NAMESPACE_END
#ifdef __cpp_exceptions
/* Tries to convert an exception ptr into its equivalent error code
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (11 commits)
File Created: July 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_UTILS_HPP
#define OUTCOME_UTILS_HPP
#include <exception>
#include <string>
#include <system_error>
OUTCOME_V2_NAMESPACE_BEGIN
#ifdef __cpp_exceptions
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
inline std::error_code error_from_exception(std::exception_ptr &&ep = std::current_exception(), std::error_code not_matched = std::make_error_code(std::errc::resource_unavailable_try_again)) noexcept
{
  if(!ep)
  {
    return {};
  }
  try
  {
    std::rethrow_exception(ep);
  }
  catch(const std::invalid_argument & /*unused*/)
  {
    ep = std::exception_ptr();
    return std::make_error_code(std::errc::invalid_argument);
  }
  catch(const std::domain_error & /*unused*/)
  {
    ep = std::exception_ptr();
    return std::make_error_code(std::errc::argument_out_of_domain);
  }
  catch(const std::length_error & /*unused*/)
  {
    ep = std::exception_ptr();
    return std::make_error_code(std::errc::argument_list_too_long);
  }
  catch(const std::out_of_range & /*unused*/)
  {
    ep = std::exception_ptr();
    return std::make_error_code(std::errc::result_out_of_range);
  }
  catch(const std::logic_error & /*unused*/) /* base class for this group */
  {
    ep = std::exception_ptr();
    return std::make_error_code(std::errc::invalid_argument);
  }
  catch(const std::system_error &e) /* also catches ios::failure */
  {
    ep = std::exception_ptr();
    return e.code();
  }
  catch(const std::overflow_error & /*unused*/)
  {
    ep = std::exception_ptr();
    return std::make_error_code(std::errc::value_too_large);
  }
  catch(const std::range_error & /*unused*/)
  {
    ep = std::exception_ptr();
    return std::make_error_code(std::errc::result_out_of_range);
  }
  catch(const std::runtime_error & /*unused*/) /* base class for this group */
  {
    ep = std::exception_ptr();
    return std::make_error_code(std::errc::resource_unavailable_try_again);
  }
  catch(const std::bad_alloc & /*unused*/)
  {
    ep = std::exception_ptr();
    return std::make_error_code(std::errc::not_enough_memory);
  }
  catch(...)
  {
  }
  return not_matched;
}
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
inline void try_throw_std_exception_from_error(std::error_code ec, const std::string &msg = std::string{})
{
  if(!ec || (ec.category() != std::generic_category()
#ifndef _WIN32
             && ec.category() != std::system_category()
#endif
             ))
  {
    return;
  }
  switch(ec.value())
  {
  case EINVAL:
    throw msg.empty() ? std::invalid_argument("invalid argument") : std::invalid_argument(msg);
  case EDOM:
    throw msg.empty() ? std::domain_error("domain error") : std::domain_error(msg);
  case E2BIG:
    throw msg.empty() ? std::length_error("length error") : std::length_error(msg);
  case ERANGE:
    throw msg.empty() ? std::out_of_range("out of range") : std::out_of_range(msg);
  case EOVERFLOW:
    throw msg.empty() ? std::overflow_error("overflow error") : std::overflow_error(msg);
  case ENOMEM:
    throw std::bad_alloc();
  }
}
#endif
OUTCOME_V2_NAMESPACE_END
#endif
OUTCOME_V2_NAMESPACE_BEGIN
namespace awaitables
{
  namespace detail
  {
    inline bool error_is_set(std::error_code ec) noexcept { return !!ec; }
    inline std::error_code error_from_exception(std::exception_ptr &&ep, std::error_code not_matched) noexcept { return OUTCOME_V2_NAMESPACE::error_from_exception(static_cast<std::exception_ptr &&>(ep), not_matched); }
  } // namespace detail
} // namespace awaitables
OUTCOME_V2_NAMESPACE_END
#endif
/* Tells C++ coroutines about Outcome's result
(C) 2019-2020 Niall Douglas <http://www.nedproductions.biz/> (12 commits)
File Created: Oct 2019


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_DETAIL_COROUTINE_SUPPORT_HPP
#define OUTCOME_DETAIL_COROUTINE_SUPPORT_HPP
#include <atomic>
#include <cassert>
#if __cpp_impl_coroutine || (defined(_MSC_VER) && __cpp_coroutines) || (defined(__clang__) && __cpp_coroutines)
#ifndef OUTCOME_HAVE_NOOP_COROUTINE
#if defined(__has_builtin)
#if __has_builtin(__builtin_coro_noop)
#define OUTCOME_HAVE_NOOP_COROUTINE 1
#endif
#endif
#endif
#ifndef OUTCOME_HAVE_NOOP_COROUTINE
#if _MSC_VER >= 1928
#define OUTCOME_HAVE_NOOP_COROUTINE 1
#else
#define OUTCOME_HAVE_NOOP_COROUTINE 0
#endif
#endif
#if __has_include(<coroutine>)
#include <coroutine>
OUTCOME_V2_NAMESPACE_BEGIN
namespace awaitables
{
  template <class Promise = void> using coroutine_handle = std::coroutine_handle<Promise>;
  template <class... Args> using coroutine_traits = std::coroutine_traits<Args...>;
  using std::suspend_always;
  using std::suspend_never;
#if OUTCOME_HAVE_NOOP_COROUTINE
  using std::noop_coroutine;
#endif
} // namespace awaitables
OUTCOME_V2_NAMESPACE_END
#define OUTCOME_FOUND_COROUTINE_HEADER 1
#elif __has_include(<experimental/coroutine>)
#include <experimental/coroutine>
OUTCOME_V2_NAMESPACE_BEGIN
namespace awaitables
{
  template <class Promise = void> using coroutine_handle = std::experimental::coroutine_handle<Promise>;
  template <class... Args> using coroutine_traits = std::experimental::coroutine_traits<Args...>;
  using std::experimental::suspend_always;
  using std::experimental::suspend_never;
#if OUTCOME_HAVE_NOOP_COROUTINE
  using std::experimental::noop_coroutine;
#endif
} // namespace awaitables
OUTCOME_V2_NAMESPACE_END
#define OUTCOME_FOUND_COROUTINE_HEADER 1
#endif
#endif
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace awaitables
{
  namespace detail
  {
    struct error_type_not_found
    {
    };
    struct exception_type_not_found
    {
    };
    template <class T> struct type_found
    {
      using type = T;
    };
    template <class T, class U = typename T::error_type> constexpr inline type_found<U> extract_error_type(int /*unused*/) { return {}; }
    template <class T> constexpr inline type_found<error_type_not_found> extract_error_type(...) { return {}; }
    template <class T, class U = typename T::exception_type> constexpr inline type_found<U> extract_exception_type(int /*unused*/) { return {}; }
    template <class T> constexpr inline type_found<exception_type_not_found> extract_exception_type(...) { return {}; }
    OUTCOME_TEMPLATE(class T, class U)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(OUTCOME_V2_NAMESPACE::detail::is_constructible<U, T>))
    inline bool try_set_error(T &&e, U *result)
    {
      new(result) U(static_cast<T &&>(e));
      return true;
    }
    template <class T> inline bool try_set_error(T && /*unused*/, ...) { return false; }
    OUTCOME_TEMPLATE(class T, class U)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(OUTCOME_V2_NAMESPACE::detail::is_constructible<U, T>))
    inline void set_or_rethrow(T &e, U *result) { new(result) U(e); }
    template <class T> inline void set_or_rethrow(T &e, ...) { rethrow_exception(e); }
    template <class T> class fake_atomic
    {
      T _v;
    public:
      constexpr fake_atomic(T v)
          : _v(v)
      {
      }
      T load(std::memory_order /*unused*/) { return _v; }
      void store(T v, std::memory_order /*unused*/) { _v = v; }
    };
#ifdef OUTCOME_FOUND_COROUTINE_HEADER
    template <class Awaitable, bool suspend_initial, bool use_atomic, bool is_void> struct outcome_promise_type
    {
      using container_type = typename Awaitable::container_type;
      using result_set_type = std::conditional_t<use_atomic, std::atomic<bool>, fake_atomic<bool>>;
      union
      {
        OUTCOME_V2_NAMESPACE::detail::empty_type _default{};
        container_type result;
      };
      result_set_type result_set{false};
      coroutine_handle<> continuation;
      outcome_promise_type() noexcept {}
      outcome_promise_type(const outcome_promise_type &) = delete;
      outcome_promise_type(outcome_promise_type &&) = delete;
      outcome_promise_type &operator=(const outcome_promise_type &) = delete;
      outcome_promise_type &operator=(outcome_promise_type &&) = delete;
      ~outcome_promise_type()
      {
        if(result_set.load(std::memory_order_acquire))
        {
          result.~container_type(); // could throw
        }
      }
      auto get_return_object()
      {
        return Awaitable{*this}; // could throw bad_alloc
      }
      void return_value(container_type &&value)
      {
        assert(!result_set.load(std::memory_order_acquire));
        if(result_set.load(std::memory_order_acquire))
        {
          result.~container_type(); // could throw
        }
        new(&result) container_type(static_cast<container_type &&>(value)); // could throw
        result_set.store(true, std::memory_order_release);
      }
      void return_value(const container_type &value)
      {
        assert(!result_set.load(std::memory_order_acquire));
        if(result_set.load(std::memory_order_acquire))
        {
          result.~container_type(); // could throw
        }
        new(&result) container_type(value); // could throw
        result_set.store(true, std::memory_order_release);
      }
      void unhandled_exception()
      {
        assert(!result_set.load(std::memory_order_acquire));
        if(result_set.load(std::memory_order_acquire))
        {
          result.~container_type();
        }
#ifdef __cpp_exceptions
        auto e = std::current_exception();
        auto ec = detail::error_from_exception(static_cast<decltype(e) &&>(e), {});
        // Try to set error code first
        if(!detail::error_is_set(ec) || !detail::try_set_error(static_cast<decltype(ec) &&>(ec), &result))
        {
          detail::set_or_rethrow(e, &result); // could throw
        }
#else
        std::terminate();
#endif
        result_set.store(true, std::memory_order_release);
      }
      auto initial_suspend() noexcept
      {
        struct awaiter
        {
          bool await_ready() noexcept { return !suspend_initial; }
          void await_resume() noexcept {}
          void await_suspend(coroutine_handle<> /*unused*/) noexcept {}
        };
        return awaiter{};
      }
      auto final_suspend() noexcept
      {
        struct awaiter
        {
          bool await_ready() noexcept { return false; }
          void await_resume() noexcept {}
#if OUTCOME_HAVE_NOOP_COROUTINE
          coroutine_handle<> await_suspend(coroutine_handle<outcome_promise_type> self) noexcept
          {
            return self.promise().continuation ? self.promise().continuation : noop_coroutine();
          }
#else
          void await_suspend(coroutine_handle<outcome_promise_type> self)
          {
            if(self.promise().continuation)
            {
              return self.promise().continuation.resume();
            }
          }
#endif
        };
        return awaiter{};
      }
    };
    template <class Awaitable, bool suspend_initial, bool use_atomic> struct outcome_promise_type<Awaitable, suspend_initial, use_atomic, true>
    {
      using container_type = void;
      using result_set_type = std::conditional_t<use_atomic, std::atomic<bool>, fake_atomic<bool>>;
      result_set_type result_set{false};
      coroutine_handle<> continuation;
      outcome_promise_type() {}
      outcome_promise_type(const outcome_promise_type &) = delete;
      outcome_promise_type(outcome_promise_type &&) = delete;
      outcome_promise_type &operator=(const outcome_promise_type &) = delete;
      outcome_promise_type &operator=(outcome_promise_type &&) = delete;
      ~outcome_promise_type() = default;
      auto get_return_object()
      {
        return Awaitable{*this}; // could throw bad_alloc
      }
      void return_void() noexcept
      {
        assert(!result_set.load(std::memory_order_acquire));
        result_set.store(true, std::memory_order_release);
      }
      void unhandled_exception()
      {
        assert(!result_set.load(std::memory_order_acquire));
        std::rethrow_exception(std::current_exception()); // throws
      }
      auto initial_suspend() noexcept
      {
        struct awaiter
        {
          bool await_ready() noexcept { return !suspend_initial; }
          void await_resume() noexcept {}
          void await_suspend(coroutine_handle<> /*unused*/) noexcept {}
        };
        return awaiter{};
      }
      auto final_suspend() noexcept
      {
        struct awaiter
        {
          bool await_ready() noexcept { return false; }
          void await_resume() noexcept {}
#if OUTCOME_HAVE_NOOP_COROUTINE
          coroutine_handle<> await_suspend(coroutine_handle<outcome_promise_type> self) noexcept
          {
            return self.promise().continuation ? self.promise().continuation : noop_coroutine();
          }
#else
          void await_suspend(coroutine_handle<outcome_promise_type> self)
          {
            if(self.promise().continuation)
            {
              return self.promise().continuation.resume();
            }
          }
#endif
        };
        return awaiter{};
      }
    };
    template <class Awaitable, bool suspend_initial, bool use_atomic>
    constexpr inline auto move_result_from_promise_if_not_void(outcome_promise_type<Awaitable, suspend_initial, use_atomic, false> &p)
    {
      return static_cast<typename Awaitable::container_type &&>(p.result);
    }
    template <class Awaitable, bool suspend_initial, bool use_atomic>
    constexpr inline void move_result_from_promise_if_not_void(outcome_promise_type<Awaitable, suspend_initial, use_atomic, true> & /*unused*/)
    {
    }
    template <class Cont, bool suspend_initial, bool use_atomic> struct OUTCOME_NODISCARD awaitable
    {
      using container_type = Cont;
      using promise_type = outcome_promise_type<awaitable, suspend_initial, use_atomic, std::is_void<container_type>::value>;
      coroutine_handle<promise_type> _h;
      awaitable(awaitable &&o) noexcept
          : _h(static_cast<coroutine_handle<promise_type> &&>(o._h))
      {
        o._h = nullptr;
      }
      awaitable(const awaitable &o) = delete;
      awaitable &operator=(awaitable &&) = delete; // as per P1056
      awaitable &operator=(const awaitable &) = delete;
      ~awaitable()
      {
        if(_h)
        {
          _h.destroy();
        }
      }
      explicit awaitable(promise_type &p) // could throw
          : _h(coroutine_handle<promise_type>::from_promise(p))
      {
      }
      bool await_ready() noexcept { return _h.promise().result_set.load(std::memory_order_acquire); }
      container_type await_resume()
      {
        assert(_h.promise().result_set.load(std::memory_order_acquire));
        if(!_h.promise().result_set.load(std::memory_order_acquire))
        {
          std::terminate();
        }
        return detail::move_result_from_promise_if_not_void(_h.promise());
      }
#if OUTCOME_HAVE_NOOP_COROUTINE
      coroutine_handle<> await_suspend(coroutine_handle<> cont) noexcept
      {
        _h.promise().continuation = cont;
        return _h;
      }
#else
      void await_suspend(coroutine_handle<> cont)
      {
        _h.promise().continuation = cont;
        _h.resume();
      }
#endif
    };
#endif
  } // namespace detail
} // namespace awaitables
OUTCOME_V2_NAMESPACE_END
#endif
#ifdef OUTCOME_FOUND_COROUTINE_HEADER
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN namespace awaitables {
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class T> using eager = OUTCOME_V2_NAMESPACE::awaitables::detail::awaitable<T, false, false>;
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class T> using atomic_eager = OUTCOME_V2_NAMESPACE::awaitables::detail::awaitable<T, false, true>;
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class T> using lazy = OUTCOME_V2_NAMESPACE::awaitables::detail::awaitable<T, true, false>;
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class T> using atomic_lazy = OUTCOME_V2_NAMESPACE::awaitables::detail::awaitable<T, true, true>;
} OUTCOME_V2_NAMESPACE_END
#endif
#undef OUTCOME_COROUTINE_SUPPORT_NAMESPACE_BEGIN
#undef OUTCOME_COROUTINE_SUPPORT_NAMESPACE_EXPORT_BEGIN
#undef OUTCOME_COROUTINE_SUPPORT_NAMESPACE_END
#endif
/* iostream specialisations for result and outcome
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (21 commits)
File Created: July 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_IOSTREAM_SUPPORT_HPP
#define OUTCOME_IOSTREAM_SUPPORT_HPP
/* A less simple result type
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (79 commits)
File Created: June 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_OUTCOME_HPP
#define OUTCOME_OUTCOME_HPP
/* A very simple result type
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (99 commits)
File Created: June 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_RESULT_HPP
#define OUTCOME_RESULT_HPP
/* A very simple result type
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (8 commits)
File Created: June 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_STD_RESULT_HPP
#define OUTCOME_STD_RESULT_HPP
/* A very simple result type
(C) 2017-2021 Niall Douglas <http://www.nedproductions.biz/> (14 commits)
File Created: June 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_BASIC_RESULT_HPP
#define OUTCOME_BASIC_RESULT_HPP
/* Says how to convert value, error and exception types
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (12 commits)
File Created: Nov 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_CONVERT_HPP
#define OUTCOME_CONVERT_HPP
/* Storage for a very simple basic_result type
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (6 commits)
File Created: Oct 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_BASIC_RESULT_STORAGE_HPP
#define OUTCOME_BASIC_RESULT_STORAGE_HPP
/* Type sugar for success and failure
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (25 commits)
File Created: July 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_SUCCESS_FAILURE_HPP
#define OUTCOME_SUCCESS_FAILURE_HPP
OUTCOME_V2_NAMESPACE_BEGIN
/*! AWAITING HUGO JSON CONVERSION TOOL
type definition template <class T> success_type. Potential doc page: `success_type<T>`
*/
template <class T> struct OUTCOME_NODISCARD success_type
{
  using value_type = T;
private:
  value_type _value;
  uint16_t _spare_storage{0};
public:
  success_type() = default;
  success_type(const success_type &) = default;
  success_type(success_type &&) = default; // NOLINT
  success_type &operator=(const success_type &) = default;
  success_type &operator=(success_type &&) = default; // NOLINT
  ~success_type() = default;
  OUTCOME_TEMPLATE(class U)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_same<success_type, std::decay_t<U>>::value))
  constexpr explicit success_type(U &&v, uint16_t spare_storage = 0)
      : _value(static_cast<U &&>(v)) // NOLINT
      , _spare_storage(spare_storage)
  {
  }
  constexpr value_type &value() & { return _value; }
  constexpr const value_type &value() const & { return _value; }
  constexpr value_type &&value() && { return static_cast<value_type &&>(_value); }
  constexpr const value_type &&value() const && { return static_cast<value_type &&>(_value); }
  constexpr uint16_t spare_storage() const { return _spare_storage; }
};
template <> struct OUTCOME_NODISCARD success_type<void>
{
  using value_type = void;
  constexpr uint16_t spare_storage() const { return 0; }
};
/*! Returns type sugar for implicitly constructing a `basic_result<T>` with a successful state,
default constructing `T` if necessary.
*/
inline constexpr success_type<void> success() noexcept
{
  return success_type<void>{};
}
/*! Returns type sugar for implicitly constructing a `basic_result<T>` with a successful state.
\effects Copies or moves the successful state supplied into the returned type sugar.
*/
template <class T> inline constexpr success_type<std::decay_t<T>> success(T &&v, uint16_t spare_storage = 0)
{
  return success_type<std::decay_t<T>>{static_cast<T &&>(v), spare_storage};
}
/*! AWAITING HUGO JSON CONVERSION TOOL
type definition template <class EC, class E = void> failure_type. Potential doc page: `failure_type<EC, EP = void>`
*/
template <class EC, class E = void> struct OUTCOME_NODISCARD failure_type
{
  using error_type = EC;
  using exception_type = E;
private:
  error_type _error;
  exception_type _exception;
  bool _have_error{false}, _have_exception{false};
  uint16_t _spare_storage{0};
  struct error_init_tag
  {
  };
  struct exception_init_tag
  {
  };
public:
  failure_type() = default;
  failure_type(const failure_type &) = default;
  failure_type(failure_type &&) = default; // NOLINT
  failure_type &operator=(const failure_type &) = default;
  failure_type &operator=(failure_type &&) = default; // NOLINT
  ~failure_type() = default;
  template <class U, class V>
  constexpr explicit failure_type(U &&u, V &&v, uint16_t spare_storage = 0)
      : _error(static_cast<U &&>(u))
      , _exception(static_cast<V &&>(v))
      , _have_error(true)
      , _have_exception(true)
      , _spare_storage(spare_storage)
  {
  }
  template <class U>
  constexpr explicit failure_type(in_place_type_t<error_type> /*unused*/, U &&u, uint16_t spare_storage = 0, error_init_tag /*unused*/ = error_init_tag())
      : _error(static_cast<U &&>(u))
      , _exception()
      , _have_error(true)
      , _spare_storage(spare_storage)
  {
  }
  template <class U>
  constexpr explicit failure_type(in_place_type_t<exception_type> /*unused*/, U &&u, uint16_t spare_storage = 0,
                                  exception_init_tag /*unused*/ = exception_init_tag())
      : _error()
      , _exception(static_cast<U &&>(u))
      , _have_exception(true)
      , _spare_storage(spare_storage)
  {
  }
  constexpr bool has_error() const { return _have_error; }
  constexpr bool has_exception() const { return _have_exception; }
  constexpr error_type &error() & { return _error; }
  constexpr const error_type &error() const & { return _error; }
  constexpr error_type &&error() && { return static_cast<error_type &&>(_error); }
  constexpr const error_type &&error() const && { return static_cast<error_type &&>(_error); }
  constexpr exception_type &exception() & { return _exception; }
  constexpr const exception_type &exception() const & { return _exception; }
  constexpr exception_type &&exception() && { return static_cast<exception_type &&>(_exception); }
  constexpr const exception_type &&exception() const && { return static_cast<exception_type &&>(_exception); }
  constexpr uint16_t spare_storage() const { return _spare_storage; }
};
template <class EC> struct OUTCOME_NODISCARD failure_type<EC, void>
{
  using error_type = EC;
  using exception_type = void;
private:
  error_type _error;
  uint16_t _spare_storage{0};
public:
  failure_type() = default;
  failure_type(const failure_type &) = default;
  failure_type(failure_type &&) = default; // NOLINT
  failure_type &operator=(const failure_type &) = default;
  failure_type &operator=(failure_type &&) = default; // NOLINT
  ~failure_type() = default;
  OUTCOME_TEMPLATE(class U)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_same<failure_type, std::decay_t<U>>::value))
  constexpr explicit failure_type(U &&u, uint16_t spare_storage = 0)
      : _error(static_cast<U &&>(u)) // NOLINT
      , _spare_storage(spare_storage)
  {
  }
  constexpr error_type &error() & { return _error; }
  constexpr const error_type &error() const & { return _error; }
  constexpr error_type &&error() && { return static_cast<error_type &&>(_error); }
  constexpr const error_type &&error() const && { return static_cast<error_type &&>(_error); }
  constexpr uint16_t spare_storage() const { return _spare_storage; }
};
template <class E> struct OUTCOME_NODISCARD failure_type<void, E>
{
  using error_type = void;
  using exception_type = E;
private:
  exception_type _exception;
  uint16_t _spare_storage{0};
public:
  failure_type() = default;
  failure_type(const failure_type &) = default;
  failure_type(failure_type &&) = default; // NOLINT
  failure_type &operator=(const failure_type &) = default;
  failure_type &operator=(failure_type &&) = default; // NOLINT
  ~failure_type() = default;
  OUTCOME_TEMPLATE(class V)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_same<failure_type, std::decay_t<V>>::value))
  constexpr explicit failure_type(V &&v, uint16_t spare_storage = 0)
      : _exception(static_cast<V &&>(v)) // NOLINT
      , _spare_storage(spare_storage)
  {
  }
  constexpr exception_type &exception() & { return _exception; }
  constexpr const exception_type &exception() const & { return _exception; }
  constexpr exception_type &&exception() && { return static_cast<exception_type &&>(_exception); }
  constexpr const exception_type &&exception() const && { return static_cast<exception_type &&>(_exception); }
  constexpr uint16_t spare_storage() const { return _spare_storage; }
};
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class EC> inline constexpr failure_type<std::decay_t<EC>> failure(EC &&v, uint16_t spare_storage = 0)
{
  return failure_type<std::decay_t<EC>>{static_cast<EC &&>(v), spare_storage};
}
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class EC, class E> inline constexpr failure_type<std::decay_t<EC>, std::decay_t<E>> failure(EC &&v, E &&w, uint16_t spare_storage = 0)
{
  return failure_type<std::decay_t<EC>, std::decay_t<E>>{static_cast<EC &&>(v), static_cast<E &&>(w), spare_storage};
}
namespace detail
{
  template <class T> struct is_success_type
  {
    static constexpr bool value = false;
  };
  template <class T> struct is_success_type<success_type<T>>
  {
    static constexpr bool value = true;
  };
  template <class T> struct is_failure_type
  {
    static constexpr bool value = false;
  };
  template <class EC, class E> struct is_failure_type<failure_type<EC, E>>
  {
    static constexpr bool value = true;
  };
} // namespace detail
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class T> static constexpr bool is_success_type = detail::is_success_type<std::decay_t<T>>::value;
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class T> static constexpr bool is_failure_type = detail::is_failure_type<std::decay_t<T>>::value;
OUTCOME_V2_NAMESPACE_END
#endif
/* Traits for Outcome
(C) 2018-2019 Niall Douglas <http://www.nedproductions.biz/> (8 commits)
File Created: March 2018


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_TRAIT_HPP
#define OUTCOME_TRAIT_HPP
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace trait
{
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class R> //
  static constexpr bool type_can_be_used_in_basic_result = //
  (!std::is_reference<R>::value //
   && !OUTCOME_V2_NAMESPACE::detail::is_in_place_type_t<std::decay_t<R>>::value //
   && !is_success_type<R> //
   && !is_failure_type<R> //
   && !std::is_array<R>::value //
   && (std::is_void<R>::value || (std::is_object<R>::value //
                                  && std::is_destructible<R>::value)) //
  );
  /*! AWAITING HUGO JSON CONVERSION TOOL
type definition  is_error_type. Potential doc page: NOT FOUND
*/
  template <class T> struct is_move_bitcopying
  {
    static constexpr bool value = false;
  };
  /*! AWAITING HUGO JSON CONVERSION TOOL
type definition  is_error_type. Potential doc page: NOT FOUND
*/
  template <class E> struct is_error_type
  {
    static constexpr bool value = false;
  };
  /*! AWAITING HUGO JSON CONVERSION TOOL
type definition  is_error_type_enum. Potential doc page: NOT FOUND
*/
  template <class E, class Enum> struct is_error_type_enum
  {
    static constexpr bool value = false;
  };
  namespace detail
  {
    template <class T> using devoid = OUTCOME_V2_NAMESPACE::detail::devoid<T>;
    template <class T> std::add_rvalue_reference_t<devoid<T>> declval() noexcept;
    // From http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4436.pdf
    namespace detector_impl
    {
      template <class...> using void_t = void;
      template <class Default, class, template <class...> class Op, class... Args> struct detector
      {
        static constexpr bool value = false;
        using type = Default;
      };
      template <class Default, template <class...> class Op, class... Args> struct detector<Default, void_t<Op<Args...>>, Op, Args...>
      {
        static constexpr bool value = true;
        using type = Op<Args...>;
      };
    } // namespace detector_impl
    template <template <class...> class Op, class... Args> using is_detected = detector_impl::detector<void, void, Op, Args...>;
    template <class Arg> using result_of_make_error_code = decltype(make_error_code(declval<Arg>()));
    template <class Arg> using introspect_make_error_code = is_detected<result_of_make_error_code, Arg>;
    template <class Arg> using result_of_make_exception_ptr = decltype(make_exception_ptr(declval<Arg>()));
    template <class Arg> using introspect_make_exception_ptr = is_detected<result_of_make_exception_ptr, Arg>;
    template <class T> struct _is_error_code_available
    {
      static constexpr bool value = detail::introspect_make_error_code<T>::value;
      using type = typename detail::introspect_make_error_code<T>::type;
    };
    template <class T> struct _is_exception_ptr_available
    {
      static constexpr bool value = detail::introspect_make_exception_ptr<T>::value;
      using type = typename detail::introspect_make_exception_ptr<T>::type;
    };
  } // namespace detail
  /*! AWAITING HUGO JSON CONVERSION TOOL
type definition  is_error_code_available. Potential doc page: NOT FOUND
*/
  template <class T> struct is_error_code_available
  {
    static constexpr bool value = detail::_is_error_code_available<std::decay_t<T>>::value;
    using type = typename detail::_is_error_code_available<std::decay_t<T>>::type;
  };
  template <class T> constexpr bool is_error_code_available_v = detail::_is_error_code_available<std::decay_t<T>>::value;
  /*! AWAITING HUGO JSON CONVERSION TOOL
type definition  is_exception_ptr_available. Potential doc page: NOT FOUND
*/
  template <class T> struct is_exception_ptr_available
  {
    static constexpr bool value = detail::_is_exception_ptr_available<std::decay_t<T>>::value;
    using type = typename detail::_is_exception_ptr_available<std::decay_t<T>>::type;
  };
  template <class T> constexpr bool is_exception_ptr_available_v = detail::_is_exception_ptr_available<std::decay_t<T>>::value;
} // namespace trait
OUTCOME_V2_NAMESPACE_END
#endif
/* Essentially an internal optional implementation :)
(C) 2017-2020 Niall Douglas <http://www.nedproductions.biz/> (24 commits)
File Created: June 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_VALUE_STORAGE_HPP
#define OUTCOME_VALUE_STORAGE_HPP
#include <cassert>
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace detail
{
  template <class T, bool nothrow> struct strong_swap_impl
  {
    constexpr strong_swap_impl(bool &allgood, T &a, T &b)
    {
      allgood = true;
      using std::swap;
      swap(a, b);
    }
  };
  template <class T, bool nothrow> struct strong_placement_impl
  {
    template <class F> constexpr strong_placement_impl(bool &allgood, T *a, T *b, F &&f)
    {
      allgood = true;
      new(a) T(static_cast<T &&>(*b));
      b->~T();
      f();
    }
  };
#ifdef __cpp_exceptions
  template <class T> struct strong_swap_impl<T, false>
  {
    strong_swap_impl(bool &allgood, T &a, T &b)
    {
      allgood = true;
      T v(static_cast<T &&>(a));
      try
      {
        a = static_cast<T &&>(b);
      }
      catch(...)
      {
        // Try to put back a
        try
        {
          a = static_cast<T &&>(v);
          // fall through as all good
        }
        catch(...)
        {
          // failed to completely restore
          allgood = false;
          // throw away second exception
        }
        throw; // rethrow original exception
      }
      // b has been moved to a, try to move v to b
      try
      {
        b = static_cast<T &&>(v);
      }
      catch(...)
      {
        // Try to restore a to b, and v to a
        try
        {
          b = static_cast<T &&>(a);
          a = static_cast<T &&>(v);
          // fall through as all good
        }
        catch(...)
        {
          // failed to completely restore
          allgood = false;
          // throw away second exception
        }
        throw; // rethrow original exception
      }
    }
  };
  template <class T> struct strong_placement_impl<T, false>
  {
    template <class F> strong_placement_impl(bool &allgood, T *a, T *b, F &&f)
    {
      new(a) T(static_cast<T &&>(*b));
      try
      {
        b->~T();
        f();
      }
      catch(...)
      {
        // Try to put back a, but only if we are still good
        if(allgood)
        {
          try
          {
            new(b) T(static_cast<T &&>(*a));
            // fall through as all good
          }
          catch(...)
          {
            // failed to completely restore
            allgood = false;
            // throw away second exception
          }
          throw; // rethrow original exception
        }
      }
    }
  };
#endif
} // namespace detail
/*!
 */
OUTCOME_TEMPLATE(class T)
OUTCOME_TREQUIRES(OUTCOME_TPRED(std::is_move_constructible<T>::value &&std::is_move_assignable<T>::value))
constexpr inline void strong_swap(bool &allgood, T &a, T &b) noexcept(detail::is_nothrow_swappable<T>::value)
{
  detail::strong_swap_impl<T, detail::is_nothrow_swappable<T>::value>(allgood, a, b);
}
/*!
 */
OUTCOME_TEMPLATE(class T, class F)
OUTCOME_TREQUIRES(OUTCOME_TPRED(std::is_move_constructible<T>::value &&std::is_move_assignable<T>::value))
constexpr inline void strong_placement(bool &allgood, T *a, T *b, F &&f) noexcept(std::is_nothrow_move_constructible<T>::value)
{
  detail::strong_placement_impl<T, std::is_nothrow_move_constructible<T>::value>(allgood, a, b, static_cast<F &&>(f));
}
namespace detail
{
  template <class T>
  constexpr
#ifdef _MSC_VER
  __declspec(noreturn)
#elif defined(__GNUC__) || defined(__clang__)
        __attribute__((noreturn))
#endif
  void make_ub(T && /*unused*/)
  {
    assert(false); // NOLINT
#if defined(__GNUC__) || defined(__clang__)
    __builtin_unreachable();
#elif defined(_MSC_VER)
    __assume(0);
#endif
  }
  /* Outcome v1 used a C bitfield whose values were tracked by compiler optimisers nicely,
  but that produces ICEs when used in constexpr.

  Outcome v2.0-v2.1 used a 32 bit integer and manually set and cleared bits. Unfortunately
  only GCC's optimiser tracks bit values during constant folding, and only per byte, and
  even then unreliably. https://wg21.link/P1886 "Error speed benchmarking" showed just how
  poorly clang and MSVC fails to optimise outcome-using code, if you manually set bits.

  Outcome v2.2 therefore uses an enum with fixed values, and constexpr manipulation functions
  to change the value to one of the enum's values. This is stupid to look at in source code,
  but it make clang's optimiser do the right thing, so it's worth it.
  */
#define OUTCOME_USE_CONSTEXPR_ENUM_STATUS 0
  enum class status : uint16_t
  {
    // WARNING: These bits are not tracked by abi-dumper, but changing them will break ABI!
    none = 0,
    have_value = (1U << 0U),
    have_error = (1U << 1U),
    have_exception = (2U << 1U),
    have_error_exception = (3U << 1U),
    // failed to complete a strong swap
    have_lost_consistency = (1U << 3U),
    have_value_lost_consistency = (1U << 0U) | (1U << 3U),
    have_error_lost_consistency = (1U << 1U) | (1U << 3U),
    have_exception_lost_consistency = (2U << 1U) | (1U << 3U),
    have_error_exception_lost_consistency = (3U << 1U) | (1U << 3U),
    // can errno be set from this error?
    have_error_is_errno = (1U << 4U),
    have_error_error_is_errno = (1U << 1U) | (1U << 4U),
    have_error_exception_error_is_errno = (3U << 1U) | (1U << 4U),
    have_error_lost_consistency_error_is_errno = (1U << 1U) | (1U << 3U) | (1U << 4U),
    have_error_exception_lost_consistency_error_is_errno = (3U << 1U) | (1U << 3U) | (1U << 4U),
    // value has been moved from
    have_moved_from = (1U << 5U)
  };
  struct status_bitfield_type
  {
    status status_value{status::none};
    uint16_t spare_storage_value{0}; // hooks::spare_storage()
    constexpr status_bitfield_type() = default;
    constexpr status_bitfield_type(status v) noexcept
        : status_value(v)
    {
    } // NOLINT
    constexpr status_bitfield_type(status v, uint16_t s) noexcept
        : status_value(v)
        , spare_storage_value(s)
    {
    }
    constexpr status_bitfield_type(const status_bitfield_type &) = default;
    constexpr status_bitfield_type(status_bitfield_type &&) = default;
    constexpr status_bitfield_type &operator=(const status_bitfield_type &) = default;
    constexpr status_bitfield_type &operator=(status_bitfield_type &&) = default;
    //~status_bitfield_type() = default;  // Do NOT uncomment this, it breaks older clangs!
    constexpr bool have_value() const noexcept
    {
      return (static_cast<uint16_t>(status_value) & static_cast<uint16_t>(status::have_value)) != 0;
    }
    constexpr bool have_error() const noexcept
    {
      return (static_cast<uint16_t>(status_value) & static_cast<uint16_t>(status::have_error)) != 0;
    }
    constexpr bool have_exception() const noexcept
    {
      return (static_cast<uint16_t>(status_value) & static_cast<uint16_t>(status::have_exception)) != 0;
    }
    constexpr bool have_lost_consistency() const noexcept
    {
      return (static_cast<uint16_t>(status_value) & static_cast<uint16_t>(status::have_lost_consistency)) != 0;
    }
    constexpr bool have_error_is_errno() const noexcept
    {
      return (static_cast<uint16_t>(status_value) & static_cast<uint16_t>(status::have_error_is_errno)) != 0;
    }
    constexpr bool have_moved_from() const noexcept
    {
      return (static_cast<uint16_t>(status_value) & static_cast<uint16_t>(status::have_moved_from)) != 0;
    }
    constexpr status_bitfield_type &set_have_value(bool v) noexcept
    {
      status_value = static_cast<status>(v ? (static_cast<uint16_t>(status_value) | static_cast<uint16_t>(status::have_value)) :
                                             (static_cast<uint16_t>(status_value) & ~static_cast<uint16_t>(status::have_value)));
      return *this;
    }
    constexpr status_bitfield_type &set_have_error(bool v) noexcept
    {
      status_value = static_cast<status>(v ? (static_cast<uint16_t>(status_value) | static_cast<uint16_t>(status::have_error)) :
                                             (static_cast<uint16_t>(status_value) & ~static_cast<uint16_t>(status::have_error)));
      return *this;
    }
    constexpr status_bitfield_type &set_have_exception(bool v) noexcept
    {
      status_value = static_cast<status>(v ? (static_cast<uint16_t>(status_value) | static_cast<uint16_t>(status::have_exception)) :
                                             (static_cast<uint16_t>(status_value) & ~static_cast<uint16_t>(status::have_exception)));
      return *this;
    }
    constexpr status_bitfield_type &set_have_error_is_errno(bool v) noexcept
    {
      status_value = static_cast<status>(v ? (static_cast<uint16_t>(status_value) | static_cast<uint16_t>(status::have_error_is_errno)) :
                                             (static_cast<uint16_t>(status_value) & ~static_cast<uint16_t>(status::have_error_is_errno)));
      return *this;
    }
    constexpr status_bitfield_type &set_have_lost_consistency(bool v) noexcept
    {
      status_value = static_cast<status>(v ? (static_cast<uint16_t>(status_value) | static_cast<uint16_t>(status::have_lost_consistency)) :
                                             (static_cast<uint16_t>(status_value) & ~static_cast<uint16_t>(status::have_lost_consistency)));
      return *this;
    }
    constexpr status_bitfield_type &set_have_moved_from(bool v) noexcept
    {
      status_value = static_cast<status>(v ? (static_cast<uint16_t>(status_value) | static_cast<uint16_t>(status::have_moved_from)) :
                                             (static_cast<uint16_t>(status_value) & ~static_cast<uint16_t>(status::have_moved_from)));
      return *this;
    }
  };
#if !defined(NDEBUG)
  // Check is trivial in all ways except default constructibility
  static_assert(sizeof(status_bitfield_type) == 4, "status_bitfield_type is not sized 4 bytes!");
  static_assert(std::is_trivially_copyable<status_bitfield_type>::value, "status_bitfield_type is not trivially copyable!");
  static_assert(std::is_trivially_assignable<status_bitfield_type, status_bitfield_type>::value, "status_bitfield_type is not trivially assignable!");
  static_assert(std::is_trivially_destructible<status_bitfield_type>::value, "status_bitfield_type is not trivially destructible!");
  static_assert(std::is_trivially_copy_constructible<status_bitfield_type>::value, "status_bitfield_type is not trivially copy constructible!");
  static_assert(std::is_trivially_move_constructible<status_bitfield_type>::value, "status_bitfield_type is not trivially move constructible!");
  static_assert(std::is_trivially_copy_assignable<status_bitfield_type>::value, "status_bitfield_type is not trivially copy assignable!");
  static_assert(std::is_trivially_move_assignable<status_bitfield_type>::value, "status_bitfield_type is not trivially move assignable!");
  // Also check is standard layout
  static_assert(std::is_standard_layout<status_bitfield_type>::value, "status_bitfield_type is not a standard layout type!");
#endif
  template <class State> constexpr inline void _set_error_is_errno(State & /*unused*/) {}
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4624) // destructor was implicitly defined as deleted
#endif
  // Used if both T and E are trivial
  template <class T, class E> struct value_storage_trivial
  {
    using value_type = T;
    using error_type = E;
    // Disable in place construction if they are the same type
    struct disable_in_place_value_type
    {
    };
    struct disable_in_place_error_type
    {
    };
    using _value_type = std::conditional_t<std::is_same<value_type, error_type>::value, disable_in_place_value_type, value_type>;
    using _error_type = std::conditional_t<std::is_same<value_type, error_type>::value, disable_in_place_error_type, error_type>;
    using _value_type_ = devoid<value_type>;
    using _error_type_ = devoid<error_type>;
    union
    {
      empty_type _empty;
      _value_type_ _value;
      _error_type_ _error;
    };
    status_bitfield_type _status;
    constexpr value_storage_trivial() noexcept
        : _empty{}
    {
    }
    value_storage_trivial(const value_storage_trivial &) = default; // NOLINT
    value_storage_trivial(value_storage_trivial &&) = default; // NOLINT
    value_storage_trivial &operator=(const value_storage_trivial &) = default; // NOLINT
    value_storage_trivial &operator=(value_storage_trivial &&) = default; // NOLINT
    ~value_storage_trivial() = default;
    constexpr explicit value_storage_trivial(status_bitfield_type status)
        : _empty()
        , _status(status)
    {
    }
    template <class... Args>
    constexpr explicit value_storage_trivial(in_place_type_t<_value_type> /*unused*/,
                                             Args &&...args) noexcept(detail::is_nothrow_constructible<_value_type_, Args...>)
        : _value(static_cast<Args &&>(args)...)
        , _status(status::have_value)
    {
    }
    template <class U, class... Args>
    constexpr value_storage_trivial(in_place_type_t<_value_type> /*unused*/, std::initializer_list<U> il,
                                    Args &&...args) noexcept(detail::is_nothrow_constructible<_value_type_, std::initializer_list<U>, Args...>)
        : _value(il, static_cast<Args &&>(args)...)
        , _status(status::have_value)
    {
    }
    template <class... Args>
    constexpr explicit value_storage_trivial(in_place_type_t<_error_type> /*unused*/,
                                             Args &&...args) noexcept(detail::is_nothrow_constructible<_error_type_, Args...>)
        : _error(static_cast<Args &&>(args)...)
        , _status(status::have_error)
    {
      _set_error_is_errno(*this);
    }
    template <class U, class... Args>
    constexpr value_storage_trivial(in_place_type_t<_error_type> /*unused*/, std::initializer_list<U> il,
                                    Args &&...args) noexcept(detail::is_nothrow_constructible<_error_type_, std::initializer_list<U>, Args...>)
        : _error(il, static_cast<Args &&>(args)...)
        , _status(status::have_error)
    {
      _set_error_is_errno(*this);
    }
    struct nonvoid_converting_constructor_tag
    {
    };
    template <class U, class V>
    static constexpr bool enable_nonvoid_converting_constructor =
    !(std::is_same<std::decay_t<U>, value_type>::value && std::is_same<std::decay_t<V>, error_type>::value) //
    && detail::is_constructible<value_type, U> && detail::is_constructible<error_type, V>;
    OUTCOME_TEMPLATE(class U, class V)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_nonvoid_converting_constructor<U, V>))
    constexpr explicit value_storage_trivial(const value_storage_trivial<U, V> &o, nonvoid_converting_constructor_tag /*unused*/ = {}) noexcept(
    detail::is_nothrow_constructible<_value_type_, U> &&detail::is_nothrow_constructible<_error_type_, V>)
        : value_storage_trivial(o._status.have_value() ?
                                value_storage_trivial(in_place_type<value_type>, o._value) :
                                (o._status.have_error() ? value_storage_trivial(in_place_type<error_type>, o._error) : value_storage_trivial())) // NOLINT
    {
      _status = o._status;
    }
    OUTCOME_TEMPLATE(class U, class V)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_nonvoid_converting_constructor<U, V>))
    constexpr explicit value_storage_trivial(value_storage_trivial<U, V> &&o, nonvoid_converting_constructor_tag /*unused*/ = {}) noexcept(
    detail::is_nothrow_constructible<_value_type_, U> &&detail::is_nothrow_constructible<_error_type_, V>)
        : value_storage_trivial(
          o._status.have_value() ?
          value_storage_trivial(in_place_type<value_type>, static_cast<U &&>(o._value)) :
          (o._status.have_error() ? value_storage_trivial(in_place_type<error_type>, static_cast<V &&>(o._error)) : value_storage_trivial())) // NOLINT
    {
      _status = o._status;
    }
    struct void_value_converting_constructor_tag
    {
    };
    template <class V>
    static constexpr bool enable_void_value_converting_constructor = std::is_default_constructible<value_type>::value &&detail::is_constructible<error_type, V>;
    OUTCOME_TEMPLATE(class V)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_void_value_converting_constructor<V>))
    constexpr explicit value_storage_trivial(const value_storage_trivial<void, V> &o, void_value_converting_constructor_tag /*unused*/ = {}) noexcept(
    std::is_nothrow_default_constructible<_value_type_>::value &&detail::is_nothrow_constructible<_error_type_, V>)
        : value_storage_trivial(o._status.have_value() ?
                                value_storage_trivial(in_place_type<value_type>) :
                                (o._status.have_error() ? value_storage_trivial(in_place_type<error_type>, o._error) : value_storage_trivial())) // NOLINT
    {
      _status = o._status;
    }
    OUTCOME_TEMPLATE(class V)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_void_value_converting_constructor<V>))
    constexpr explicit value_storage_trivial(value_storage_trivial<void, V> &&o, void_value_converting_constructor_tag /*unused*/ = {}) noexcept(
    std::is_nothrow_default_constructible<_value_type_>::value &&detail::is_nothrow_constructible<_error_type_, V>)
        : value_storage_trivial(
          o._status.have_value() ?
          value_storage_trivial(in_place_type<value_type>) :
          (o._status.have_error() ? value_storage_trivial(in_place_type<error_type>, static_cast<V &&>(o._error)) : value_storage_trivial())) // NOLINT
    {
      _status = o._status;
    }
    struct void_error_converting_constructor_tag
    {
    };
    template <class U>
    static constexpr bool enable_void_error_converting_constructor = std::is_default_constructible<error_type>::value &&detail::is_constructible<value_type, U>;
    OUTCOME_TEMPLATE(class U)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_void_error_converting_constructor<U>))
    constexpr explicit value_storage_trivial(const value_storage_trivial<U, void> &o, void_error_converting_constructor_tag /*unused*/ = {}) noexcept(
    detail::is_nothrow_constructible<_value_type_, U> &&std::is_nothrow_default_constructible<_error_type_>::value)
        : value_storage_trivial(o._status.have_value() ?
                                value_storage_trivial(in_place_type<value_type>, o._value) :
                                (o._status.have_error() ? value_storage_trivial(in_place_type<error_type>) : value_storage_trivial())) // NOLINT
    {
      _status = o._status;
    }
    OUTCOME_TEMPLATE(class U)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_void_error_converting_constructor<U>))
    constexpr explicit value_storage_trivial(value_storage_trivial<U, void> &&o, void_error_converting_constructor_tag /*unused*/ = {}) noexcept(
    detail::is_nothrow_constructible<_value_type_, U> &&std::is_nothrow_default_constructible<_error_type_>::value)
        : value_storage_trivial(o._status.have_value() ?
                                value_storage_trivial(in_place_type<value_type>, static_cast<U &&>(o._value)) :
                                (o._status.have_error() ? value_storage_trivial(in_place_type<error_type>) : value_storage_trivial())) // NOLINT
    {
      _status = o._status;
    }
    constexpr void swap(value_storage_trivial &o) noexcept
    {
      // storage is trivial, so just use assignment
      auto temp = static_cast<value_storage_trivial &&>(*this);
      *this = static_cast<value_storage_trivial &&>(o);
      o = static_cast<value_storage_trivial &&>(temp);
    }
  };
  /* Used if T or E is non-trivial. The additional constexpr is injected in C++ 20 to enable Outcome to
  work in constexpr evaluation contexts in C++ 20 where non-trivial constexpr destructors are now allowed.
  */
  template <class T, class E> struct value_storage_nontrivial
  {
    using value_type = T;
    using error_type = E;
    struct disable_in_place_value_type
    {
    };
    struct disable_in_place_error_type
    {
    };
    using _value_type = std::conditional_t<std::is_same<value_type, error_type>::value, disable_in_place_value_type, value_type>;
    using _error_type = std::conditional_t<std::is_same<value_type, error_type>::value, disable_in_place_error_type, error_type>;
    using _value_type_ = devoid<value_type>;
    using _error_type_ = devoid<error_type>;
    union
    {
      empty_type _empty1;
      _value_type_ _value;
    };
    status_bitfield_type _status;
    union
    {
      empty_type _empty2;
      _error_type_ _error;
    };
#if __cplusplus >= 202000 || _HAS_CXX20
    constexpr
#endif
    value_storage_nontrivial() noexcept
        : _empty1{}
        , _empty2{}
    {
    }
    value_storage_nontrivial &operator=(const value_storage_nontrivial &) = default; // if reaches here, copy assignment is trivial
    value_storage_nontrivial &operator=(value_storage_nontrivial &&) = default; // NOLINT if reaches here, move assignment is trivial
#if __cplusplus >= 202000 || _HAS_CXX20
    constexpr
#endif
    value_storage_nontrivial(value_storage_nontrivial &&o) noexcept(
    std::is_nothrow_move_constructible<_value_type_>::value &&std::is_nothrow_move_constructible<_error_type_>::value) // NOLINT
    {
      if(o._status.have_value())
      {
        new(&_value) _value_type_(static_cast<_value_type_ &&>(o._value)); // NOLINT
      }
      else if(o._status.have_error())
      {
        new(&_error) _error_type_(static_cast<_error_type_ &&>(o._error)); // NOLINT
      }
      _status = o._status;
      o._status.set_have_moved_from(true);
    }
#if __cplusplus >= 202000 || _HAS_CXX20
    constexpr
#endif
    value_storage_nontrivial(const value_storage_nontrivial &o) noexcept(
    std::is_nothrow_copy_constructible<_value_type_>::value &&std::is_nothrow_copy_constructible<_error_type_>::value)
    {
      if(o._status.have_value())
      {
        new(&_value) _value_type_(o._value); // NOLINT
      }
      else if(o._status.have_error())
      {
        new(&_error) _error_type_(o._error); // NOLINT
      }
      _status = o._status;
    }
#if __cplusplus >= 202000 || _HAS_CXX20
    constexpr
#endif
    explicit value_storage_nontrivial(status_bitfield_type status)
        : _empty1()
        , _status(status)
        , _empty2()
    {
    }
    template <class... Args>
    constexpr explicit value_storage_nontrivial(in_place_type_t<_value_type> /*unused*/,
                                                Args &&...args) noexcept(detail::is_nothrow_constructible<_value_type_, Args...>)
        : _value(static_cast<Args &&>(args)...) // NOLINT
        , _status(status::have_value)
    {
    }
    template <class U, class... Args>
    constexpr value_storage_nontrivial(in_place_type_t<_value_type> /*unused*/, std::initializer_list<U> il,
                                       Args &&...args) noexcept(detail::is_nothrow_constructible<_value_type_, std::initializer_list<U>, Args...>)
        : _value(il, static_cast<Args &&>(args)...)
        , _status(status::have_value)
    {
    }
    template <class... Args>
    constexpr explicit value_storage_nontrivial(in_place_type_t<_error_type> /*unused*/,
                                                Args &&...args) noexcept(detail::is_nothrow_constructible<_error_type_, Args...>)
        : _status(status::have_error)
        , _error(static_cast<Args &&>(args)...) // NOLINT
    {
      _set_error_is_errno(*this);
    }
    template <class U, class... Args>
    constexpr value_storage_nontrivial(in_place_type_t<_error_type> /*unused*/, std::initializer_list<U> il,
                                       Args &&...args) noexcept(detail::is_nothrow_constructible<_error_type_, std::initializer_list<U>, Args...>)
        : _status(status::have_error)
        , _error(il, static_cast<Args &&>(args)...)
    {
      _set_error_is_errno(*this);
    }
    struct nonvoid_converting_constructor_tag
    {
    };
    template <class U, class V>
    static constexpr bool enable_nonvoid_converting_constructor =
    !(std::is_same<std::decay_t<U>, value_type>::value && std::is_same<std::decay_t<V>, error_type>::value) //
    && detail::is_constructible<value_type, U> && detail::is_constructible<error_type, V>;
    OUTCOME_TEMPLATE(class U, class V)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_nonvoid_converting_constructor<U, V>))
    constexpr explicit value_storage_nontrivial(const value_storage_trivial<U, V> &o, nonvoid_converting_constructor_tag /*unused*/ = {}) noexcept(
    detail::is_nothrow_constructible<_value_type_, U> &&detail::is_nothrow_constructible<_error_type_, V>)
        : value_storage_nontrivial(o._status.have_value() ?
                                   value_storage_nontrivial(in_place_type<value_type>, o._value) :
                                   (o._status.have_error() ? value_storage_nontrivial(in_place_type<error_type>, o._error) : value_storage_nontrivial()))
    {
      _status = o._status;
    }
    OUTCOME_TEMPLATE(class U, class V)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_nonvoid_converting_constructor<U, V>))
    constexpr explicit value_storage_nontrivial(value_storage_trivial<U, V> &&o, nonvoid_converting_constructor_tag /*unused*/ = {}) noexcept(
    detail::is_nothrow_constructible<_value_type_, U> &&detail::is_nothrow_constructible<_error_type_, V>)
        : value_storage_nontrivial(
          o._status.have_value() ?
          value_storage_nontrivial(in_place_type<value_type>, static_cast<U &&>(o._value)) :
          (o._status.have_error() ? value_storage_nontrivial(in_place_type<error_type>, static_cast<V &&>(o._error)) : value_storage_nontrivial()))
    {
      _status = o._status;
    }
    OUTCOME_TEMPLATE(class U, class V)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_nonvoid_converting_constructor<U, V>))
    constexpr explicit value_storage_nontrivial(const value_storage_nontrivial<U, V> &o, nonvoid_converting_constructor_tag /*unused*/ = {}) noexcept(
    detail::is_nothrow_constructible<_value_type_, U> &&detail::is_nothrow_constructible<_error_type_, V>)
        : value_storage_nontrivial(o._status.have_value() ?
                                   value_storage_nontrivial(in_place_type<value_type>, o._value) :
                                   (o._status.have_error() ? value_storage_nontrivial(in_place_type<error_type>, o._error) : value_storage_nontrivial()))
    {
      _status = o._status;
    }
    OUTCOME_TEMPLATE(class U, class V)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_nonvoid_converting_constructor<U, V>))
    constexpr explicit value_storage_nontrivial(value_storage_nontrivial<U, V> &&o, nonvoid_converting_constructor_tag /*unused*/ = {}) noexcept(
    detail::is_nothrow_constructible<_value_type_, U> &&detail::is_nothrow_constructible<_error_type_, V>)
        : value_storage_nontrivial(
          o._status.have_value() ?
          value_storage_nontrivial(in_place_type<value_type>, static_cast<U &&>(o._value)) :
          (o._status.have_error() ? value_storage_nontrivial(in_place_type<error_type>, static_cast<V &&>(o._error)) : value_storage_nontrivial()))
    {
      _status = o._status;
    }
    struct void_value_converting_constructor_tag
    {
    };
    template <class V>
    static constexpr bool enable_void_value_converting_constructor = std::is_default_constructible<value_type>::value &&detail::is_constructible<error_type, V>;
    OUTCOME_TEMPLATE(class V)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_void_value_converting_constructor<V>))
    constexpr explicit value_storage_nontrivial(const value_storage_trivial<void, V> &o, void_value_converting_constructor_tag /*unused*/ = {}) noexcept(
    std::is_nothrow_default_constructible<_value_type_>::value &&detail::is_nothrow_constructible<_error_type_, V>)
    {
      if(o._status.have_value())
      {
        new(&_value) _value_type_(); // NOLINT
      }
      else if(o._status.have_error())
      {
        new(&_error) _error_type_(o._error); // NOLINT
      }
      _status = o._status;
    }
    OUTCOME_TEMPLATE(class V)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_void_value_converting_constructor<V>))
    constexpr explicit value_storage_nontrivial(value_storage_trivial<void, V> &&o, void_value_converting_constructor_tag /*unused*/ = {}) noexcept(
    std::is_nothrow_default_constructible<_value_type_>::value &&detail::is_nothrow_constructible<_error_type_, V>)
    {
      if(o._status.have_value())
      {
        new(&_value) _value_type_(); // NOLINT
      }
      else if(o._status.have_error())
      {
        new(&_error) _error_type_(static_cast<_error_type_ &&>(o._error)); // NOLINT
      }
      _status = o._status;
      o._status.set_have_moved_from(true);
    }
    struct void_error_converting_constructor_tag
    {
    };
    template <class U>
    static constexpr bool enable_void_error_converting_constructor = std::is_default_constructible<error_type>::value &&detail::is_constructible<value_type, U>;
    OUTCOME_TEMPLATE(class U)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_void_error_converting_constructor<U>))
    constexpr explicit value_storage_nontrivial(const value_storage_trivial<U, void> &o, void_error_converting_constructor_tag /*unused*/ = {}) noexcept(
    detail::is_nothrow_constructible<_value_type_, U> &&std::is_nothrow_default_constructible<_error_type_>::value)
    {
      if(o._status.have_value())
      {
        new(&_value) _value_type_(o._value); // NOLINT
      }
      else if(o._status.have_error())
      {
        new(&_error) _error_type_(); // NOLINT
      }
      _status = o._status;
    }
    OUTCOME_TEMPLATE(class U)
    OUTCOME_TREQUIRES(OUTCOME_TPRED(enable_void_error_converting_constructor<U>))
    constexpr explicit value_storage_nontrivial(value_storage_trivial<U, void> &&o, void_error_converting_constructor_tag /*unused*/ = {}) noexcept(
    detail::is_nothrow_constructible<_value_type_, U> &&std::is_nothrow_default_constructible<_error_type_>::value)
    {
      if(o._status.have_value())
      {
        new(&_value) _value_type_(static_cast<_value_type_ &&>(o._value)); // NOLINT
      }
      else if(o._status.have_error())
      {
        new(&_error) _error_type_(); // NOLINT
      }
      _status = o._status;
      o._status.set_have_moved_from(true);
    }
#if __cplusplus >= 202000 || _HAS_CXX20
    constexpr
#endif
    ~value_storage_nontrivial() noexcept(std::is_nothrow_destructible<_value_type_>::value &&std::is_nothrow_destructible<_error_type_>::value)
    {
      if(this->_status.have_value())
      {
        if(!trait::is_move_bitcopying<value_type>::value || !this->_status.have_moved_from())
        {
          this->_value.~_value_type_(); // NOLINT
        }
        this->_status.set_have_value(false);
      }
      else if(this->_status.have_error())
      {
        if(!trait::is_move_bitcopying<error_type>::value || !this->_status.have_moved_from())
        {
          this->_error.~_error_type_(); // NOLINT
        }
        this->_status.set_have_error(false);
      }
    }
#if __cplusplus >= 202000 || _HAS_CXX20
    constexpr
#endif
    void
    swap(value_storage_nontrivial &o) noexcept(detail::is_nothrow_swappable<_value_type_>::value &&detail::is_nothrow_swappable<_error_type_>::value)
    {
      using std::swap;
      // empty/empty
      if(!_status.have_value() && !o._status.have_value() && !_status.have_error() && !o._status.have_error())
      {
        swap(_status, o._status);
        return;
      }
      // value/value
      if(_status.have_value() && o._status.have_value())
      {
        struct _
        {
          status_bitfield_type &a, &b;
          bool all_good{false};
          ~_()
          {
            if(!all_good)
            {
              // We lost one of the values
              a.set_have_lost_consistency(true);
              b.set_have_lost_consistency(true);
            }
          }
        } _{_status, o._status};
        strong_swap(_.all_good, _value, o._value);
        swap(_status, o._status);
        return;
      }
      // error/error
      if(_status.have_error() && o._status.have_error())
      {
        struct _
        {
          status_bitfield_type &a, &b;
          bool all_good{false};
          ~_()
          {
            if(!all_good)
            {
              // We lost one of the values
              a.set_have_lost_consistency(true);
              b.set_have_lost_consistency(true);
            }
          }
        } _{_status, o._status};
        strong_swap(_.all_good, _error, o._error);
        swap(_status, o._status);
        return;
      }
      // Could be value/empty, error/empty, etc
      if(_status.have_value() && !o._status.have_error())
      {
        // Move construct me into other
        new(&o._value) _value_type_(static_cast<_value_type_ &&>(_value)); // NOLINT
        if(!trait::is_move_bitcopying<value_type>::value)
        {
          this->_value.~value_type(); // NOLINT
        }
        swap(_status, o._status);
        return;
      }
      if(o._status.have_value() && !_status.have_error())
      {
        // Move construct other into me
        new(&_value) _value_type_(static_cast<_value_type_ &&>(o._value)); // NOLINT
        if(!trait::is_move_bitcopying<value_type>::value)
        {
          o._value.~value_type(); // NOLINT
        }
        swap(_status, o._status);
        return;
      }
      if(_status.have_error() && !o._status.have_value())
      {
        // Move construct me into other
        new(&o._error) _error_type_(static_cast<_error_type_ &&>(_error)); // NOLINT
        if(!trait::is_move_bitcopying<error_type>::value)
        {
          this->_error.~error_type(); // NOLINT
        }
        swap(_status, o._status);
        return;
      }
      if(o._status.have_error() && !_status.have_value())
      {
        // Move construct other into me
        new(&_error) _error_type_(static_cast<_error_type_ &&>(o._error)); // NOLINT
        if(!trait::is_move_bitcopying<error_type>::value)
        {
          o._error.~error_type(); // NOLINT
        }
        swap(_status, o._status);
        return;
      }
      // It can now only be value/error, or error/value
      struct _
      {
        status_bitfield_type &a, &b;
        _value_type_ *value, *o_value;
        _error_type_ *error, *o_error;
        bool all_good{true};
        ~_()
        {
          if(!all_good)
          {
            // We lost one of the values
            a.set_have_lost_consistency(true);
            b.set_have_lost_consistency(true);
          }
        }
      } _{_status, o._status, &_value, &o._value, &_error, &o._error};
      if(_status.have_value() && o._status.have_error())
      {
        strong_placement(_.all_good, _.o_value, _.value, [&_] { //
          strong_placement(_.all_good, _.error, _.o_error, [&_] { //
            swap(_.a, _.b); //
          });
        });
        return;
      }
      if(_status.have_error() && o._status.have_value())
      {
        strong_placement(_.all_good, _.o_error, _.error, [&_] { //
          strong_placement(_.all_good, _.value, _.o_value, [&_] { //
            swap(_.a, _.b); //
          });
        });
        return;
      }
      // Should never reach here
      make_ub(_value);
    }
  };
  template <class Base> struct value_storage_delete_copy_constructor : Base // NOLINT
  {
    using Base::Base;
    using value_type = typename Base::value_type;
    using error_type = typename Base::error_type;
    value_storage_delete_copy_constructor() = default;
    value_storage_delete_copy_constructor(const value_storage_delete_copy_constructor &) = delete;
    value_storage_delete_copy_constructor(value_storage_delete_copy_constructor &&) = default; // NOLINT
  };
  template <class Base> struct value_storage_delete_copy_assignment : Base // NOLINT
  {
    using Base::Base;
    using value_type = typename Base::value_type;
    using error_type = typename Base::error_type;
    value_storage_delete_copy_assignment() = default;
    value_storage_delete_copy_assignment(const value_storage_delete_copy_assignment &) = default;
    value_storage_delete_copy_assignment(value_storage_delete_copy_assignment &&) = default; // NOLINT
    value_storage_delete_copy_assignment &operator=(const value_storage_delete_copy_assignment &o) = delete;
    value_storage_delete_copy_assignment &operator=(value_storage_delete_copy_assignment &&o) = default; // NOLINT
  };
  template <class Base> struct value_storage_delete_move_assignment : Base // NOLINT
  {
    using Base::Base;
    using value_type = typename Base::value_type;
    using error_type = typename Base::error_type;
    value_storage_delete_move_assignment() = default;
    value_storage_delete_move_assignment(const value_storage_delete_move_assignment &) = default;
    value_storage_delete_move_assignment(value_storage_delete_move_assignment &&) = default; // NOLINT
    value_storage_delete_move_assignment &operator=(const value_storage_delete_move_assignment &o) = default;
    value_storage_delete_move_assignment &operator=(value_storage_delete_move_assignment &&o) = delete;
  };
  template <class Base> struct value_storage_delete_move_constructor : Base // NOLINT
  {
    using Base::Base;
    using value_type = typename Base::value_type;
    using error_type = typename Base::error_type;
    value_storage_delete_move_constructor() = default;
    value_storage_delete_move_constructor(const value_storage_delete_move_constructor &) = default;
    value_storage_delete_move_constructor(value_storage_delete_move_constructor &&) = delete;
  };
  template <class Base> struct value_storage_nontrivial_move_assignment : Base // NOLINT
  {
    using Base::Base;
    using value_type = typename Base::value_type;
    using error_type = typename Base::error_type;
    value_storage_nontrivial_move_assignment() = default;
    value_storage_nontrivial_move_assignment(const value_storage_nontrivial_move_assignment &) = default;
    value_storage_nontrivial_move_assignment(value_storage_nontrivial_move_assignment &&) = default; // NOLINT
    value_storage_nontrivial_move_assignment &operator=(const value_storage_nontrivial_move_assignment &o) = default;
#if __cplusplus >= 202000 || _HAS_CXX20
    constexpr
#endif
    value_storage_nontrivial_move_assignment &
    operator=(value_storage_nontrivial_move_assignment &&o) noexcept(
    std::is_nothrow_move_assignable<value_type>::value &&std::is_nothrow_move_assignable<error_type>::value) // NOLINT
    {
      using _value_type_ = typename Base::_value_type_;
      using _error_type_ = typename Base::_error_type_;
      if(!this->_status.have_value() && !this->_status.have_error() && !o._status.have_value() && !o._status.have_error())
      {
        this->_status = o._status;
        o._status.set_have_moved_from(true);
        return *this;
      }
      if(this->_status.have_value() && o._status.have_value())
      {
        this->_value = static_cast<_value_type_ &&>(o._value); // NOLINT
        this->_status = o._status;
        o._status.set_have_moved_from(true);
        return *this;
      }
      if(this->_status.have_error() && o._status.have_error())
      {
        this->_error = static_cast<_error_type_ &&>(o._error); // NOLINT
        this->_status = o._status;
        o._status.set_have_moved_from(true);
        return *this;
      }
      if(this->_status.have_value() && !o._status.have_value() && !o._status.have_error())
      {
        if(!trait::is_move_bitcopying<value_type>::value || this->_status.have_moved_from())
        {
          this->_value.~_value_type_(); // NOLINT
        }
        this->_status = o._status;
        o._status.set_have_moved_from(true);
        return *this;
      }
      if(!this->_status.have_value() && !this->_status.have_error() && o._status.have_value())
      {
        new(&this->_value) _value_type_(static_cast<_value_type_ &&>(o._value)); // NOLINT
        this->_status = o._status;
        o._status.set_have_moved_from(true);
        return *this;
      }
      if(this->_status.have_error() && !o._status.have_value() && !o._status.have_error())
      {
        if(!trait::is_move_bitcopying<error_type>::value || this->_status.have_moved_from())
        {
          this->_error.~_error_type_(); // NOLINT
        }
        this->_status = o._status;
        o._status.set_have_moved_from(true);
        return *this;
      }
      if(!this->_status.have_value() && !this->_status.have_error() && o._status.have_error())
      {
        new(&this->_error) _error_type_(static_cast<_error_type_ &&>(o._error)); // NOLINT
        this->_status = o._status;
        o._status.set_have_moved_from(true);
        return *this;
      }
      if(this->_status.have_value() && o._status.have_error())
      {
        if(!trait::is_move_bitcopying<value_type>::value || this->_status.have_moved_from())
        {
          this->_value.~_value_type_(); // NOLINT
        }
        new(&this->_error) _error_type_(static_cast<_error_type_ &&>(o._error)); // NOLINT
        this->_status = o._status;
        o._status.set_have_moved_from(true);
        return *this;
      }
      if(this->_status.have_error() && o._status.have_value())
      {
        if(!trait::is_move_bitcopying<error_type>::value || this->_status.have_moved_from())
        {
          this->_error.~_error_type_(); // NOLINT
        }
        new(&this->_value) _value_type_(static_cast<_value_type_ &&>(o._value)); // NOLINT
        this->_status = o._status;
        o._status.set_have_moved_from(true);
        return *this;
      }
      // Should never reach here
      make_ub(this->_value);
    }
  };
  template <class Base> struct value_storage_nontrivial_copy_assignment : Base // NOLINT
  {
    using Base::Base;
    using value_type = typename Base::value_type;
    using error_type = typename Base::error_type;
    value_storage_nontrivial_copy_assignment() = default;
    value_storage_nontrivial_copy_assignment(const value_storage_nontrivial_copy_assignment &) = default;
    value_storage_nontrivial_copy_assignment(value_storage_nontrivial_copy_assignment &&) = default; // NOLINT
    value_storage_nontrivial_copy_assignment &operator=(value_storage_nontrivial_copy_assignment &&o) = default; // NOLINT
#if __cplusplus >= 202000 || _HAS_CXX20
    constexpr
#endif
    value_storage_nontrivial_copy_assignment &
    operator=(const value_storage_nontrivial_copy_assignment &o) noexcept(
    std::is_nothrow_copy_assignable<value_type>::value &&std::is_nothrow_copy_assignable<error_type>::value)
    {
      using _value_type_ = typename Base::_value_type_;
      using _error_type_ = typename Base::_error_type_;
      if(!this->_status.have_value() && !this->_status.have_error() && !o._status.have_value() && !o._status.have_error())
      {
        this->_status = o._status;
        return *this;
      }
      if(this->_status.have_value() && o._status.have_value())
      {
        this->_value = o._value; // NOLINT
        this->_status = o._status;
        return *this;
      }
      if(this->_status.have_error() && o._status.have_error())
      {
        this->_error = o._error; // NOLINT
        this->_status = o._status;
        return *this;
      }
      if(this->_status.have_value() && !o._status.have_value() && !o._status.have_error())
      {
        if(!trait::is_move_bitcopying<value_type>::value || this->_status.have_moved_from())
        {
          this->_value.~_value_type_(); // NOLINT
        }
        this->_status = o._status;
        return *this;
      }
      if(!this->_status.have_value() && !this->_status.have_error() && o._status.have_value())
      {
        new(&this->_value) _value_type_(o._value); // NOLINT
        this->_status = o._status;
        return *this;
      }
      if(this->_status.have_error() && !o._status.have_value() && !o._status.have_error())
      {
        if(!trait::is_move_bitcopying<error_type>::value || this->_status.have_moved_from())
        {
          this->_error.~_error_type_(); // NOLINT
        }
        this->_status = o._status;
        return *this;
      }
      if(!this->_status.have_value() && !this->_status.have_error() && o._status.have_error())
      {
        new(&this->_error) _error_type_(o._error); // NOLINT
        this->_status = o._status;
        return *this;
      }
      if(this->_status.have_value() && o._status.have_error())
      {
        if(!trait::is_move_bitcopying<value_type>::value || this->_status.have_moved_from())
        {
          this->_value.~_value_type_(); // NOLINT
        }
        new(&this->_error) _error_type_(o._error); // NOLINT
        this->_status = o._status;
        return *this;
      }
      if(this->_status.have_error() && o._status.have_value())
      {
        if(!trait::is_move_bitcopying<error_type>::value || this->_status.have_moved_from())
        {
          this->_error.~_error_type_(); // NOLINT
        }
        new(&this->_value) _value_type_(o._value); // NOLINT
        this->_status = o._status;
        return *this;
      }
      // Should never reach here
      make_ub(this->_value);
    }
  };
#ifdef _MSC_VER
#pragma warning(pop)
#endif
  // is_trivially_copyable is true even if type is not copyable, so handle that here
  template <class T> struct is_storage_trivial
  {
    static constexpr bool value = std::is_void<T>::value || (std::is_trivially_copy_constructible<T>::value && std::is_trivially_copyable<T>::value);
  };
  // work around libstdc++ 7 bug
  template <> struct is_storage_trivial<void>
  {
    static constexpr bool value = true;
  };
  template <> struct is_storage_trivial<const void>
  {
    static constexpr bool value = true;
  };
  template <class T, class E>
  using value_storage_select_trivality =
  std::conditional_t<is_storage_trivial<T>::value && is_storage_trivial<E>::value, value_storage_trivial<T, E>, value_storage_nontrivial<T, E>>;
  template <class T, class E>
  using value_storage_select_move_constructor =
  std::conditional_t<std::is_move_constructible<devoid<T>>::value && std::is_move_constructible<devoid<E>>::value, value_storage_select_trivality<T, E>,
                     value_storage_delete_move_constructor<value_storage_select_trivality<T, E>>>;
  template <class T, class E>
  using value_storage_select_copy_constructor =
  std::conditional_t<std::is_copy_constructible<devoid<T>>::value && std::is_copy_constructible<devoid<E>>::value, value_storage_select_move_constructor<T, E>,
                     value_storage_delete_copy_constructor<value_storage_select_move_constructor<T, E>>>;
  template <class T, class E>
  using value_storage_select_move_assignment =
  std::conditional_t<std::is_trivially_move_assignable<devoid<T>>::value && std::is_trivially_move_assignable<devoid<E>>::value,
                     value_storage_select_copy_constructor<T, E>,
                     std::conditional_t<std::is_move_assignable<devoid<T>>::value && std::is_move_assignable<devoid<E>>::value,
                                        value_storage_nontrivial_move_assignment<value_storage_select_copy_constructor<T, E>>,
                                        value_storage_delete_copy_assignment<value_storage_select_copy_constructor<T, E>>>>;
  template <class T, class E>
  using value_storage_select_copy_assignment =
  std::conditional_t<std::is_trivially_copy_assignable<devoid<T>>::value && std::is_trivially_copy_assignable<devoid<E>>::value,
                     value_storage_select_move_assignment<T, E>,
                     std::conditional_t<std::is_copy_assignable<devoid<T>>::value && std::is_copy_assignable<devoid<E>>::value,
                                        value_storage_nontrivial_copy_assignment<value_storage_select_move_assignment<T, E>>,
                                        value_storage_delete_copy_assignment<value_storage_select_move_assignment<T, E>>>>;
  template <class T, class E> using value_storage_select_impl = value_storage_select_copy_assignment<T, E>;
#ifndef NDEBUG
  // Check is trivial in all ways except default constructibility
  // static_assert(std::is_trivial<value_storage_select_impl<int, long>>::value, "value_storage_select_impl<int, long> is not trivial!");
  // static_assert(std::is_trivially_default_constructible<value_storage_select_impl<int, long>>::value, "value_storage_select_impl<int, long> is not trivially
  // default constructible!");
  static_assert(std::is_trivially_copyable<value_storage_select_impl<int, long>>::value, "value_storage_select_impl<int, long> is not trivially copyable!");
  static_assert(std::is_trivially_assignable<value_storage_select_impl<int, long>, value_storage_select_impl<int, long>>::value,
                "value_storage_select_impl<int, long> is not trivially assignable!");
  static_assert(std::is_trivially_destructible<value_storage_select_impl<int, long>>::value,
                "value_storage_select_impl<int, long> is not trivially destructible!");
  static_assert(std::is_trivially_copy_constructible<value_storage_select_impl<int, long>>::value,
                "value_storage_select_impl<int, long> is not trivially copy constructible!");
  static_assert(std::is_trivially_move_constructible<value_storage_select_impl<int, long>>::value,
                "value_storage_select_impl<int, long> is not trivially move constructible!");
  static_assert(std::is_trivially_copy_assignable<value_storage_select_impl<int, long>>::value,
                "value_storage_select_impl<int, long> is not trivially copy assignable!");
  static_assert(std::is_trivially_move_assignable<value_storage_select_impl<int, long>>::value,
                "value_storage_select_impl<int, long> is not trivially move assignable!");
  // Also check is standard layout
  static_assert(std::is_standard_layout<value_storage_select_impl<int, long>>::value, "value_storage_select_impl<int, long> is not a standard layout type!");
#endif
} // namespace detail
OUTCOME_V2_NAMESPACE_END
#endif
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace detail
{
  template <class R, class EC, class NoValuePolicy> class basic_result_storage;
} // namespace detail
namespace hooks
{
  template <class R, class S, class NoValuePolicy> constexpr inline uint16_t spare_storage(const detail::basic_result_storage<R, S, NoValuePolicy> *r) noexcept;
  template <class R, class S, class NoValuePolicy>
  constexpr inline void set_spare_storage(detail::basic_result_storage<R, S, NoValuePolicy> *r, uint16_t v) noexcept;
} // namespace hooks
namespace policy
{
  struct base;
} // namespace policy
namespace detail
{
  template <class R, class EC, class NoValuePolicy> //
  class basic_result_storage
  {
    static_assert(trait::type_can_be_used_in_basic_result<R>, "The type R cannot be used in a basic_result");
    static_assert(trait::type_can_be_used_in_basic_result<EC>, "The type S cannot be used in a basic_result");
    friend struct policy::base;
    template <class T, class U, class V> //
    friend class basic_result_storage;
    template <class T, class U, class V> friend class basic_result_final;
    template <class T, class U, class V>
    friend constexpr inline uint16_t hooks::spare_storage(const detail::basic_result_storage<T, U, V> *r) noexcept; // NOLINT
    template <class T, class U, class V>
    friend constexpr inline void hooks::set_spare_storage(detail::basic_result_storage<T, U, V> *r, uint16_t v) noexcept; // NOLINT
    struct disable_in_place_value_type
    {
    };
    struct disable_in_place_error_type
    {
    };
  protected:
    using _value_type = std::conditional_t<std::is_same<R, EC>::value, disable_in_place_value_type, R>;
    using _error_type = std::conditional_t<std::is_same<R, EC>::value, disable_in_place_error_type, EC>;
    using _state_type = value_storage_select_impl<_value_type, _error_type>;
    _state_type _state;
  public:
    // Used by iostream support to access state
    _state_type &_iostreams_state() { return _state; }
    const _state_type &_iostreams_state() const { return _state; }
  protected:
    basic_result_storage() = default;
    basic_result_storage(const basic_result_storage &) = default; // NOLINT
    basic_result_storage(basic_result_storage &&) = default; // NOLINT
    basic_result_storage &operator=(const basic_result_storage &) = default; // NOLINT
    basic_result_storage &operator=(basic_result_storage &&) = default; // NOLINT
    ~basic_result_storage() = default;
    template <class... Args>
    constexpr explicit basic_result_storage(in_place_type_t<_value_type> _,
                                            Args &&... args) noexcept(detail::is_nothrow_constructible<_value_type, Args...>)
        : _state{_, static_cast<Args &&>(args)...}
    {
    }
    template <class U, class... Args>
    constexpr basic_result_storage(in_place_type_t<_value_type> _, std::initializer_list<U> il,
                                   Args &&... args) noexcept(detail::is_nothrow_constructible<_value_type, std::initializer_list<U>, Args...>)
        : _state{_, il, static_cast<Args &&>(args)...}
    {
    }
    template <class... Args>
    constexpr explicit basic_result_storage(in_place_type_t<_error_type> _,
                                            Args &&... args) noexcept(detail::is_nothrow_constructible<_error_type, Args...>)
        : _state{_, static_cast<Args &&>(args)...}
    {
    }
    template <class U, class... Args>
    constexpr basic_result_storage(in_place_type_t<_error_type> _, std::initializer_list<U> il,
                                   Args &&... args) noexcept(detail::is_nothrow_constructible<_error_type, std::initializer_list<U>, Args...>)
        : _state{_, il, static_cast<Args &&>(args)...}
    {
    }
    struct compatible_conversion_tag
    {
    };
    template <class T, class U, class V>
    constexpr basic_result_storage(compatible_conversion_tag /*unused*/, const basic_result_storage<T, U, V> &o) noexcept(
    detail::is_nothrow_constructible<_value_type, T> &&detail::is_nothrow_constructible<_error_type, U>)
        : _state(o._state)
    {
    }
    template <class T, class U, class V>
    constexpr basic_result_storage(compatible_conversion_tag /*unused*/, basic_result_storage<T, U, V> &&o) noexcept(
    detail::is_nothrow_constructible<_value_type, T> &&detail::is_nothrow_constructible<_error_type, U>)
        : _state(static_cast<decltype(o._state) &&>(o._state))
    {
    }
    struct make_error_code_compatible_conversion_tag
    {
    };
    template <class T, class U, class V>
    constexpr basic_result_storage(make_error_code_compatible_conversion_tag /*unused*/, const basic_result_storage<T, U, V> &o) noexcept(
    detail::is_nothrow_constructible<_value_type, T> &&noexcept(make_error_code(std::declval<U>())))
        : _state(o._state._status.have_value() ? _state_type(in_place_type<_value_type>, o._state._value) :
                                                 _state_type(in_place_type<_error_type>, make_error_code(o._state._error)))
    {
    }
    template <class T, class U, class V>
    constexpr basic_result_storage(make_error_code_compatible_conversion_tag /*unused*/, basic_result_storage<T, U, V> &&o) noexcept(
    detail::is_nothrow_constructible<_value_type, T> &&noexcept(make_error_code(std::declval<U>())))
        : _state(o._state._status.have_value() ? _state_type(in_place_type<_value_type>, static_cast<T &&>(o._state._value)) :
                                                 _state_type(in_place_type<_error_type>, make_error_code(static_cast<U &&>(o._state._error))))
    {
    }
    struct make_exception_ptr_compatible_conversion_tag
    {
    };
    template <class T, class U, class V>
    constexpr basic_result_storage(make_exception_ptr_compatible_conversion_tag /*unused*/, const basic_result_storage<T, U, V> &o) noexcept(
    detail::is_nothrow_constructible<_value_type, T> &&noexcept(make_exception_ptr(std::declval<U>())))
        : _state(o._state._status.have_value() ? _state_type(in_place_type<_value_type>, o._state._value) :
                                                 _state_type(in_place_type<_error_type>, make_exception_ptr(o._state._error)))
    {
    }
    template <class T, class U, class V>
    constexpr basic_result_storage(make_exception_ptr_compatible_conversion_tag /*unused*/, basic_result_storage<T, U, V> &&o) noexcept(
    detail::is_nothrow_constructible<_value_type, T> &&noexcept(make_exception_ptr(std::declval<U>())))
        : _state(o._state._status.have_value() ? _state_type(in_place_type<_value_type>, static_cast<T &&>(o._state._value)) :
                                                 _state_type(in_place_type<_error_type>, make_exception_ptr(static_cast<U &&>(o._state._error))))
    {
    }
  };
} // namespace detail
OUTCOME_V2_NAMESPACE_END
#endif
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace concepts
{
#if defined(__cpp_concepts)
#if (defined(_MSC_VER) || defined(__clang__) || (defined(__GNUC__) && __cpp_concepts >= 201707) || OUTCOME_FORCE_STD_CXX_CONCEPTS) && !OUTCOME_FORCE_LEGACY_GCC_CXX_CONCEPTS
#define OUTCOME_GCC6_CONCEPT_BOOL
#else
#define OUTCOME_GCC6_CONCEPT_BOOL bool
#endif
  namespace detail
  {
    template <class T, class U> concept OUTCOME_GCC6_CONCEPT_BOOL SameHelper = std::is_same<T, U>::value;
    template <class T, class U> concept OUTCOME_GCC6_CONCEPT_BOOL same_as = detail::SameHelper<T, U> &&detail::SameHelper<U, T>;
    template <class T, class U> concept OUTCOME_GCC6_CONCEPT_BOOL convertible = std::is_convertible<T, U>::value;
    template <class T, class U> concept OUTCOME_GCC6_CONCEPT_BOOL base_of = std::is_base_of<T, U>::value;
  } // namespace detail
  /* The `value_or_none` concept.
  \requires That `U::value_type` exists and that `std::declval<U>().has_value()` returns a `bool` and `std::declval<U>().value()` exists.
  */
  template <class U> concept OUTCOME_GCC6_CONCEPT_BOOL value_or_none = requires(U a)
  {
    {
      a.has_value()
    }
    ->detail::same_as<bool>;
    {a.value()};
  };
  /* The `value_or_error` concept.
  \requires That `U::value_type` and `U::error_type` exist;
  that `std::declval<U>().has_value()` returns a `bool`, `std::declval<U>().value()` and  `std::declval<U>().error()` exists.
  */
  template <class U> concept OUTCOME_GCC6_CONCEPT_BOOL value_or_error = requires(U a)
  {
    {
      a.has_value()
    }
    ->detail::same_as<bool>;
    {a.value()};
    {a.error()};
  };
#else
  namespace detail
  {
    struct no_match
    {
    };
    inline no_match match_value_or_none(...);
    inline no_match match_value_or_error(...);
    OUTCOME_TEMPLATE(class U)
    OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<U>().has_value()), OUTCOME_TEXPR(std::declval<U>().value()))
    inline U match_value_or_none(U &&);
    OUTCOME_TEMPLATE(class U)
    OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<U>().has_value()), OUTCOME_TEXPR(std::declval<U>().value()), OUTCOME_TEXPR(std::declval<U>().error()))
    inline U match_value_or_error(U &&);
    template <class U>
    static constexpr bool value_or_none =
    !std::is_same<no_match, decltype(match_value_or_none(std::declval<OUTCOME_V2_NAMESPACE::detail::devoid<U>>()))>::value;
    template <class U>
    static constexpr bool value_or_error =
    !std::is_same<no_match, decltype(match_value_or_error(std::declval<OUTCOME_V2_NAMESPACE::detail::devoid<U>>()))>::value;
  } // namespace detail
  /* The `value_or_none` concept.
  \requires That `U::value_type` exists and that `std::declval<U>().has_value()` returns a `bool` and `std::declval<U>().value()` exists.
  */
  template <class U> static constexpr bool value_or_none = detail::value_or_none<U>;
  /* The `value_or_error` concept.
  \requires That `U::value_type` and `U::error_type` exist;
  that `std::declval<U>().has_value()` returns a `bool`, `std::declval<U>().value()` and  `std::declval<U>().error()` exists.
  */
  template <class U> static constexpr bool value_or_error = detail::value_or_error<U>;
#endif
} // namespace concepts
namespace convert
{
#if OUTCOME_ENABLE_LEGACY_SUPPORT_FOR < 220
#if defined(__cpp_concepts)
  template <class U> concept OUTCOME_GCC6_CONCEPT_BOOL ValueOrNone = concepts::value_or_none<U>;
  template <class U> concept OUTCOME_GCC6_CONCEPT_BOOL ValueOrError = concepts::value_or_error<U>;
#else
  template <class U> static constexpr bool ValueOrNone = concepts::value_or_none<U>;
  template <class U> static constexpr bool ValueOrError = concepts::value_or_error<U>;
#endif
#endif
  namespace detail
  {
    template <class T, class X> struct make_type
    {
      template <class U> static constexpr T value(U &&v) { return T{in_place_type<typename T::value_type>, static_cast<U &&>(v).value()}; }
      template <class U> static constexpr T error(U &&v) { return T{in_place_type<typename T::error_type>, static_cast<U &&>(v).error()}; }
      static constexpr T error() { return T{in_place_type<typename T::error_type>}; }
    };
    template <class T> struct make_type<T, void>
    {
      template <class U> static constexpr T value(U && /*unused*/) { return T{in_place_type<typename T::value_type>}; }
      template <class U> static constexpr T error(U && /*unused*/) { return T{in_place_type<typename T::error_type>}; }
      static constexpr T error() { return T{in_place_type<typename T::error_type>}; }
    };
  } // namespace detail
  /*! AWAITING HUGO JSON CONVERSION TOOL
type definition  value_or_error. Potential doc page: NOT FOUND
*/
  template <class T, class U> struct value_or_error
  {
    static constexpr bool enable_result_inputs = false;
    static constexpr bool enable_outcome_inputs = false;
    OUTCOME_TEMPLATE(class X)
    OUTCOME_TREQUIRES(
    OUTCOME_TPRED(std::is_same<U, std::decay_t<X>>::value //
                  &&concepts::value_or_error<U> //
                  && (std::is_void<typename std::decay_t<X>::value_type>::value ||
                      OUTCOME_V2_NAMESPACE::detail::is_explicitly_constructible<typename T::value_type, typename std::decay_t<X>::value_type>) //
                  &&(std::is_void<typename std::decay_t<X>::error_type>::value ||
                     OUTCOME_V2_NAMESPACE::detail::is_explicitly_constructible<typename T::error_type, typename std::decay_t<X>::error_type>) ))
    constexpr T operator()(X &&v)
    {
      return v.has_value() ? detail::make_type<T, typename T::value_type>::value(static_cast<X &&>(v)) :
                             detail::make_type<T, typename U::error_type>::error(static_cast<X &&>(v));
    }
  };
} // namespace convert
OUTCOME_V2_NAMESPACE_END
#endif
/* Finaliser for a very simple result type
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (5 commits)
File Created: Oct 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_BASIC_RESULT_FINAL_HPP
#define OUTCOME_BASIC_RESULT_FINAL_HPP
/* Error observers for a very simple basic_result type
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (2 commits)
File Created: Oct 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_BASIC_RESULT_ERROR_OBSERVERS_HPP
#define OUTCOME_BASIC_RESULT_ERROR_OBSERVERS_HPP
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace detail
{
  template <class Base, class EC, class NoValuePolicy> class basic_result_error_observers : public Base
  {
  public:
    using error_type = EC;
    using Base::Base;
    constexpr error_type &assume_error() & noexcept
    {
      NoValuePolicy::narrow_error_check(static_cast<basic_result_error_observers &>(*this));
      return this->_state._error;
    }
    constexpr const error_type &assume_error() const &noexcept
    {
      NoValuePolicy::narrow_error_check(static_cast<const basic_result_error_observers &>(*this));
      return this->_state._error;
    }
    constexpr error_type &&assume_error() && noexcept
    {
      NoValuePolicy::narrow_error_check(static_cast<basic_result_error_observers &&>(*this));
      return static_cast<error_type &&>(this->_state._error);
    }
    constexpr const error_type &&assume_error() const &&noexcept
    {
      NoValuePolicy::narrow_error_check(static_cast<const basic_result_error_observers &&>(*this));
      return static_cast<const error_type &&>(this->_state._error);
    }
    constexpr error_type &error() &
    {
      NoValuePolicy::wide_error_check(static_cast<basic_result_error_observers &>(*this));
      return this->_state._error;
    }
    constexpr const error_type &error() const &
    {
      NoValuePolicy::wide_error_check(static_cast<const basic_result_error_observers &>(*this));
      return this->_state._error;
    }
    constexpr error_type &&error() &&
    {
      NoValuePolicy::wide_error_check(static_cast<basic_result_error_observers &&>(*this));
      return static_cast<error_type &&>(this->_state._error);
    }
    constexpr const error_type &&error() const &&
    {
      NoValuePolicy::wide_error_check(static_cast<const basic_result_error_observers &&>(*this));
      return static_cast<const error_type &&>(this->_state._error);
    }
  };
  template <class Base, class NoValuePolicy> class basic_result_error_observers<Base, void, NoValuePolicy> : public Base
  {
  public:
    using Base::Base;
    constexpr void assume_error() const noexcept { NoValuePolicy::narrow_error_check(*this); }
    constexpr void error() const { NoValuePolicy::wide_error_check(*this); }
  };
} // namespace detail
OUTCOME_V2_NAMESPACE_END
#endif
/* Value observers for a very simple basic_result type
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (2 commits)
File Created: Oct 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_RESULT_VALUE_OBSERVERS_HPP
#define OUTCOME_RESULT_VALUE_OBSERVERS_HPP
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace detail
{
  template <class Base, class R, class NoValuePolicy> class basic_result_value_observers : public Base
  {
  public:
    using value_type = R;
    using Base::Base;
    constexpr value_type &assume_value() & noexcept
    {
      NoValuePolicy::narrow_value_check(static_cast<basic_result_value_observers &>(*this));
      return this->_state._value; // NOLINT
    }
    constexpr const value_type &assume_value() const &noexcept
    {
      NoValuePolicy::narrow_value_check(static_cast<const basic_result_value_observers &>(*this));
      return this->_state._value; // NOLINT
    }
    constexpr value_type &&assume_value() && noexcept
    {
      NoValuePolicy::narrow_value_check(static_cast<basic_result_value_observers &&>(*this));
      return static_cast<value_type &&>(this->_state._value); // NOLINT
    }
    constexpr const value_type &&assume_value() const &&noexcept
    {
      NoValuePolicy::narrow_value_check(static_cast<const basic_result_value_observers &&>(*this));
      return static_cast<const value_type &&>(this->_state._value); // NOLINT
    }
    constexpr value_type &value() &
    {
      NoValuePolicy::wide_value_check(static_cast<basic_result_value_observers &>(*this));
      return this->_state._value; // NOLINT
    }
    constexpr const value_type &value() const &
    {
      NoValuePolicy::wide_value_check(static_cast<const basic_result_value_observers &>(*this));
      return this->_state._value; // NOLINT
    }
    constexpr value_type &&value() &&
    {
      NoValuePolicy::wide_value_check(static_cast<basic_result_value_observers &&>(*this));
      return static_cast<value_type &&>(this->_state._value); // NOLINT
    }
    constexpr const value_type &&value() const &&
    {
      NoValuePolicy::wide_value_check(static_cast<const basic_result_value_observers &&>(*this));
      return static_cast<const value_type &&>(this->_state._value); // NOLINT
    }
  };
  template <class Base, class NoValuePolicy> class basic_result_value_observers<Base, void, NoValuePolicy> : public Base
  {
  public:
    using Base::Base;
    constexpr void assume_value() const noexcept { NoValuePolicy::narrow_value_check(*this); }
    constexpr void value() const { NoValuePolicy::wide_value_check(*this); }
  };
} // namespace detail
OUTCOME_V2_NAMESPACE_END
#endif
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace detail
{
  template <class R, class EC, class NoValuePolicy> using select_basic_result_impl = basic_result_error_observers<basic_result_value_observers<basic_result_storage<R, EC, NoValuePolicy>, R, NoValuePolicy>, EC, NoValuePolicy>;
  template <class R, class S, class NoValuePolicy>
  class basic_result_final
  : public select_basic_result_impl<R, S, NoValuePolicy>
  {
    using base = select_basic_result_impl<R, S, NoValuePolicy>;
  public:
    using base::base;
    constexpr explicit operator bool() const noexcept { return this->_state._status.have_value(); }
    constexpr bool has_value() const noexcept { return this->_state._status.have_value(); }
    constexpr bool has_error() const noexcept { return this->_state._status.have_error(); }
    constexpr bool has_exception() const noexcept { return this->_state._status.have_exception(); }
    constexpr bool has_lost_consistency() const noexcept { return this->_state._status.have_lost_consistency(); }
    constexpr bool has_failure() const noexcept { return this->_state._status.have_error() || this->_state._status.have_exception(); }
    OUTCOME_TEMPLATE(class T, class U, class V)
    OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<detail::devoid<R>>() == std::declval<detail::devoid<T>>()), //
                      OUTCOME_TEXPR(std::declval<detail::devoid<S>>() == std::declval<detail::devoid<U>>()))
    constexpr bool operator==(const basic_result_final<T, U, V> &o) const noexcept( //
    noexcept(std::declval<detail::devoid<R>>() == std::declval<detail::devoid<T>>()) && noexcept(std::declval<detail::devoid<S>>() == std::declval<detail::devoid<U>>()))
    {
      if(this->_state._status.have_value() && o._state._status.have_value())
      {
        return this->_state._value == o._state._value; // NOLINT
      }
      if(this->_state._status.have_error() && o._state._status.have_error())
      {
        return this->_state._error == o._state._error;
      }
      return false;
    }
    OUTCOME_TEMPLATE(class T)
    OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<R>() == std::declval<T>()))
    constexpr bool operator==(const success_type<T> &o) const noexcept( //
    noexcept(std::declval<R>() == std::declval<T>()))
    {
      if(this->_state._status.have_value())
      {
        return this->_state._value == o.value();
      }
      return false;
    }
    constexpr bool operator==(const success_type<void> &o) const noexcept
    {
      (void) o;
      return this->_state._status.have_value();
    }
    OUTCOME_TEMPLATE(class T)
    OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<S>() == std::declval<T>()))
    constexpr bool operator==(const failure_type<T, void> &o) const noexcept( //
    noexcept(std::declval<S>() == std::declval<T>()))
    {
      if(this->_state._status.have_error())
      {
        return this->_state._error == o.error();
      }
      return false;
    }
    OUTCOME_TEMPLATE(class T, class U, class V)
    OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<detail::devoid<R>>() != std::declval<detail::devoid<T>>()), //
                      OUTCOME_TEXPR(std::declval<detail::devoid<S>>() != std::declval<detail::devoid<U>>()))
    constexpr bool operator!=(const basic_result_final<T, U, V> &o) const noexcept( //
    noexcept(std::declval<detail::devoid<R>>() != std::declval<detail::devoid<T>>()) && noexcept(std::declval<detail::devoid<S>>() != std::declval<detail::devoid<U>>()))
    {
      if(this->_state._status.have_value() && o._state._status.have_value())
      {
        return this->_state._value != o._state._value;
      }
      if(this->_state._status.have_error() && o._state._status.have_error())
      {
        return this->_state._error != o._state._error;
      }
      return true;
    }
    OUTCOME_TEMPLATE(class T)
    OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<R>() != std::declval<T>()))
    constexpr bool operator!=(const success_type<T> &o) const noexcept( //
    noexcept(std::declval<R>() != std::declval<T>()))
    {
      if(this->_state._status.have_value())
      {
        return this->_state._value != o.value();
      }
      return false;
    }
    constexpr bool operator!=(const success_type<void> &o) const noexcept
    {
      (void) o;
      return !this->_state._status.have_value();
    }
    OUTCOME_TEMPLATE(class T)
    OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<S>() != std::declval<T>()))
    constexpr bool operator!=(const failure_type<T, void> &o) const noexcept( //
    noexcept(std::declval<S>() != std::declval<T>()))
    {
      if(this->_state._status.have_error())
      {
        return this->_state._error != o.error();
      }
      return true;
    }
  };
  template <class T, class U, class V, class W> constexpr inline bool operator==(const success_type<W> &a, const basic_result_final<T, U, V> &b) noexcept(noexcept(b == a)) { return b == a; }
  template <class T, class U, class V, class W> constexpr inline bool operator==(const failure_type<W, void> &a, const basic_result_final<T, U, V> &b) noexcept(noexcept(b == a)) { return b == a; }
  template <class T, class U, class V, class W> constexpr inline bool operator!=(const success_type<W> &a, const basic_result_final<T, U, V> &b) noexcept(noexcept(b == a)) { return b != a; }
  template <class T, class U, class V, class W> constexpr inline bool operator!=(const failure_type<W, void> &a, const basic_result_final<T, U, V> &b) noexcept(noexcept(b == a)) { return b != a; }
} // namespace detail
OUTCOME_V2_NAMESPACE_END
#endif
/* Policies for result and outcome
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (13 commits)
File Created: Oct 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_POLICY_ALL_NARROW_HPP
#define OUTCOME_POLICY_ALL_NARROW_HPP
/* Policies for result and outcome
(C) 2017-2020 Niall Douglas <http://www.nedproductions.biz/> (6 commits) and Andrzej Krzemieński <akrzemi1@gmail.com> (1 commit)
File Created: Oct 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_POLICY_BASE_HPP
#define OUTCOME_POLICY_BASE_HPP
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
#if OUTCOME_ENABLE_LEGACY_SUPPORT_FOR < 220
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
namespace hooks
{
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class T, class U> constexpr inline void hook_result_construction(T * /*unused*/, U && /*unused*/) noexcept {}
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class T, class U> constexpr inline void hook_result_copy_construction(T * /*unused*/, U && /*unused*/) noexcept {}
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class T, class U> constexpr inline void hook_result_move_construction(T * /*unused*/, U && /*unused*/) noexcept {}
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class T, class U, class... Args>
  constexpr inline void hook_result_in_place_construction(T * /*unused*/, in_place_type_t<U> /*unused*/, Args &&... /*unused*/) noexcept
  {
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class T, class... U> constexpr inline void hook_outcome_construction(T * /*unused*/, U &&... /*unused*/) noexcept {}
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class T, class U> constexpr inline void hook_outcome_copy_construction(T * /*unused*/, U && /*unused*/) noexcept {}
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class T, class U> constexpr inline void hook_outcome_move_construction(T * /*unused*/, U && /*unused*/) noexcept {}
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class T, class U, class... Args>
  constexpr inline void hook_outcome_in_place_construction(T * /*unused*/, in_place_type_t<U> /*unused*/, Args &&... /*unused*/) noexcept
  {
  }
} // namespace hooks
#endif
namespace policy
{
  namespace detail
  {
    using OUTCOME_V2_NAMESPACE::detail::make_ub;
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  struct base
  {
    template <class... Args> static constexpr void _silence_unused(Args &&... /*unused*/) noexcept {}
  protected:
    template <class Impl> static constexpr void _make_ub(Impl &&self) noexcept { return detail::make_ub(static_cast<Impl &&>(self)); }
    template <class Impl> static constexpr bool _has_value(Impl &&self) noexcept { return self._state._status.have_value(); }
    template <class Impl> static constexpr bool _has_error(Impl &&self) noexcept { return self._state._status.have_error(); }
    template <class Impl> static constexpr bool _has_exception(Impl &&self) noexcept { return self._state._status.have_exception(); }
    template <class Impl> static constexpr bool _has_error_is_errno(Impl &&self) noexcept { return self._state._status.have_error_is_errno(); }
    template <class Impl> static constexpr void _set_has_value(Impl &&self, bool v) noexcept { self._state._status.set_have_value(v); }
    template <class Impl> static constexpr void _set_has_error(Impl &&self, bool v) noexcept { self._state._status.set_have_error(v); }
    template <class Impl> static constexpr void _set_has_exception(Impl &&self, bool v) noexcept { self._state._status.set_have_exception(v); }
    template <class Impl> static constexpr void _set_has_error_is_errno(Impl &&self, bool v) noexcept { self._state._status.set_have_error_is_errno(v); }
    template <class Impl> static constexpr auto &&_value(Impl &&self) noexcept { return static_cast<Impl &&>(self)._state._value; }
    template <class Impl> static constexpr auto &&_error(Impl &&self) noexcept { return static_cast<Impl &&>(self)._state._error; }
  public:
    template <class R, class S, class P, class NoValuePolicy, class Impl> static inline constexpr auto &&_exception(Impl &&self) noexcept;
    template <class T, class U> static constexpr inline void on_result_construction(T *inst, U &&v) noexcept
    {
#if OUTCOME_ENABLE_LEGACY_SUPPORT_FOR < 220
      using namespace hooks;
      hook_result_construction(inst, static_cast<U &&>(v));
#else
      (void) inst;
      (void) v;
#endif
    }
    template <class T, class U> static constexpr inline void on_result_copy_construction(T *inst, U &&v) noexcept
    {
#if OUTCOME_ENABLE_LEGACY_SUPPORT_FOR < 220
      using namespace hooks;
      hook_result_copy_construction(inst, static_cast<U &&>(v));
#else
      (void) inst;
      (void) v;
#endif
    }
    template <class T, class U> static constexpr inline void on_result_move_construction(T *inst, U &&v) noexcept
    {
#if OUTCOME_ENABLE_LEGACY_SUPPORT_FOR < 220
      using namespace hooks;
      hook_result_move_construction(inst, static_cast<U &&>(v));
#else
      (void) inst;
      (void) v;
#endif
    }
    template <class T, class U, class... Args>
    static constexpr inline void on_result_in_place_construction(T *inst, in_place_type_t<U> _, Args &&... args) noexcept
    {
#if OUTCOME_ENABLE_LEGACY_SUPPORT_FOR < 220
      using namespace hooks;
      hook_result_in_place_construction(inst, _, static_cast<Args &&>(args)...);
#else
      (void) inst;
      (void) _;
      _silence_unused(static_cast<Args &&>(args)...);
#endif
    }
    template <class T, class... U> static constexpr inline void on_outcome_construction(T *inst, U &&... args) noexcept
    {
#if OUTCOME_ENABLE_LEGACY_SUPPORT_FOR < 220
      using namespace hooks;
      hook_outcome_construction(inst, static_cast<U &&>(args)...);
#else
      (void) inst;
      _silence_unused(static_cast<U &&>(args)...);
#endif
    }
    template <class T, class U> static constexpr inline void on_outcome_copy_construction(T *inst, U &&v) noexcept
    {
#if OUTCOME_ENABLE_LEGACY_SUPPORT_FOR < 220
      using namespace hooks;
      hook_outcome_copy_construction(inst, static_cast<U &&>(v));
#else
      (void) inst;
      (void) v;
#endif
    }
    template <class T, class U> static constexpr inline void on_outcome_move_construction(T *inst, U &&v) noexcept
    {
#if OUTCOME_ENABLE_LEGACY_SUPPORT_FOR < 220
      using namespace hooks;
      hook_outcome_move_construction(inst, static_cast<U &&>(v));
#else
      (void) inst;
      (void) v;
#endif
    }
    template <class T, class U, class... Args>
    static constexpr inline void on_outcome_in_place_construction(T *inst, in_place_type_t<U> _, Args &&... args) noexcept
    {
#if OUTCOME_ENABLE_LEGACY_SUPPORT_FOR < 220
      using namespace hooks;
      hook_outcome_in_place_construction(inst, _, static_cast<Args &&>(args)...);
#else
      (void) inst;
      (void) _;
      _silence_unused(static_cast<Args &&>(args)...);
#endif
    }
    template <class Impl> static constexpr void narrow_value_check(Impl &&self) noexcept
    {
      if(!_has_value(self))
      {
        _make_ub(self);
      }
    }
    template <class Impl> static constexpr void narrow_error_check(Impl &&self) noexcept
    {
      if(!_has_error(self))
      {
        _make_ub(self);
      }
    }
    template <class Impl> static constexpr void narrow_exception_check(Impl &&self) noexcept
    {
      if(!_has_exception(self))
      {
        _make_ub(self);
      }
    }
  };
} // namespace policy
OUTCOME_V2_NAMESPACE_END
#endif
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace policy
{
  /*! AWAITING HUGO JSON CONVERSION TOOL
type definition  all_narrow. Potential doc page: `all_narrow`
*/
  struct all_narrow : base
  {
    template <class Impl> static constexpr void wide_value_check(Impl &&self) { base::narrow_value_check(static_cast<Impl &&>(self)); }
    template <class Impl> static constexpr void wide_error_check(Impl &&self) { base::narrow_error_check(static_cast<Impl &&>(self)); }
    template <class Impl> static constexpr void wide_exception_check(Impl &&self) { base::narrow_exception_check(static_cast<Impl &&>(self)); }
  };
} // namespace policy
OUTCOME_V2_NAMESPACE_END
#endif
/* Policies for result and outcome
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (12 commits)
File Created: Oct 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_POLICY_TERMINATE_HPP
#define OUTCOME_POLICY_TERMINATE_HPP
#include <cstdlib>
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace policy
{
  /*! AWAITING HUGO JSON CONVERSION TOOL
type definition  terminate. Potential doc page: `terminate`
*/
  struct terminate : base
  {
    template <class Impl> static constexpr void wide_value_check(Impl &&self)
    {
      if(!base::_has_value(static_cast<Impl &&>(self)))
      {
        std::abort();
      }
    }
    template <class Impl> static constexpr void wide_error_check(Impl &&self) noexcept
    {
      if(!base::_has_error(static_cast<Impl &&>(self)))
      {
        std::abort();
      }
    }
    template <class Impl> static constexpr void wide_exception_check(Impl &&self)
    {
      if(!base::_has_exception(static_cast<Impl &&>(self)))
      {
        std::abort();
      }
    }
  };
} // namespace policy
OUTCOME_V2_NAMESPACE_END
#endif
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation" // Standardese markup confuses clang
#endif
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
template <class R, class S, class NoValuePolicy> //
class basic_result;
namespace detail
{
  // These are reused by basic_outcome to save load on the compiler
  template <class value_type, class error_type> struct result_predicates
  {
    // Predicate for the implicit constructors to be available. Weakened to allow result<int, C enum>.
    static constexpr bool implicit_constructors_enabled = //
    !(trait::is_error_type<std::decay_t<value_type>>::value &&
      trait::is_error_type<std::decay_t<error_type>>::value) // both value and error types are not whitelisted error types
    && ((!detail::is_implicitly_constructible<value_type, error_type> &&
         !detail::is_implicitly_constructible<error_type, value_type>) // if value and error types cannot be constructed into one another
        || (trait::is_error_type<std::decay_t<error_type>>::value // if error type is a whitelisted error type
            && !detail::is_implicitly_constructible<error_type, value_type> // AND which cannot be constructed from the value type
            && std::is_integral<value_type>::value)); // AND the value type is some integral type
    // Predicate for the value converting constructor to be available. Weakened to allow result<int, C enum>.
    template <class T>
    static constexpr bool enable_value_converting_constructor = //
    implicit_constructors_enabled //
    && !is_in_place_type_t<std::decay_t<T>>::value // not in place construction
    && !trait::is_error_type_enum<error_type, std::decay_t<T>>::value // not an enum valid for my error type
    && ((detail::is_implicitly_constructible<value_type, T> && !detail::is_implicitly_constructible<error_type, T>) // is unambiguously for value type
        || (std::is_same<value_type, std::decay_t<T>>::value // OR is my value type exactly
            && detail::is_implicitly_constructible<value_type, T>) ); // and my value type is constructible from this ref form of T
    // Predicate for the error converting constructor to be available. Weakened to allow result<int, C enum>.
    template <class T>
    static constexpr bool enable_error_converting_constructor = //
    implicit_constructors_enabled //
    && !is_in_place_type_t<std::decay_t<T>>::value // not in place construction
    && !trait::is_error_type_enum<error_type, std::decay_t<T>>::value // not an enum valid for my error type
    && ((!detail::is_implicitly_constructible<value_type, T> && detail::is_implicitly_constructible<error_type, T>) // is unambiguously for error type
        || (std::is_same<error_type, std::decay_t<T>>::value // OR is my error type exactly
            && detail::is_implicitly_constructible<error_type, T>) ); // and my error type is constructible from this ref form of T
    // Predicate for the error condition converting constructor to be available.
    template <class ErrorCondEnum>
    static constexpr bool enable_error_condition_converting_constructor = //
    !is_in_place_type_t<std::decay_t<ErrorCondEnum>>::value // not in place construction
    && trait::is_error_type_enum<error_type, std::decay_t<ErrorCondEnum>>::value // is an error condition enum
    /*&& !detail::is_implicitly_constructible<value_type, ErrorCondEnum> && !detail::is_implicitly_constructible<error_type, ErrorCondEnum>*/; // not
                                                                                                                                                // constructible
                                                                                                                                                // via any other
                                                                                                                                                // means
    // Predicate for the converting constructor from a compatible input to be available.
    template <class T, class U, class V>
    static constexpr bool enable_compatible_conversion = //
    (std::is_void<T>::value ||
     detail::is_explicitly_constructible<value_type, typename basic_result<T, U, V>::value_type>) // if our value types are constructible
    &&(std::is_void<U>::value ||
       detail::is_explicitly_constructible<error_type, typename basic_result<T, U, V>::error_type>) // if our error types are constructible
    ;
    // Predicate for the converting constructor from a make_error_code() of the input to be available.
    template <class T, class U, class V>
    static constexpr bool enable_make_error_code_compatible_conversion = //
    trait::is_error_code_available<std::decay_t<error_type>>::value // if error type has an error code
    && !enable_compatible_conversion<T, U, V> // and the normal compatible conversion is not available
    && (std::is_void<T>::value ||
        detail::is_explicitly_constructible<value_type, typename basic_result<T, U, V>::value_type>) // and if our value types are constructible
    &&detail::is_explicitly_constructible<error_type,
                                          typename trait::is_error_code_available<U>::type>; // and our error type is constructible from a make_error_code()
    // Predicate for the converting constructor from a make_exception_ptr() of the input to be available.
    template <class T, class U, class V>
    static constexpr bool enable_make_exception_ptr_compatible_conversion = //
    trait::is_exception_ptr_available<std::decay_t<error_type>>::value // if error type has an exception ptr
    && !enable_compatible_conversion<T, U, V> // and the normal compatible conversion is not available
    && (std::is_void<T>::value ||
        detail::is_explicitly_constructible<value_type, typename basic_result<T, U, V>::value_type>) // and if our value types are constructible
    &&detail::is_explicitly_constructible<error_type, typename trait::is_exception_ptr_available<U>::type>; // and our error type is constructible from a
                                                                                                             // make_exception_ptr()
    // Predicate for the implicit converting inplace constructor from a compatible input to be available.
    struct disable_inplace_value_error_constructor;
    template <class... Args>
    using choose_inplace_value_error_constructor = std::conditional_t< //
    detail::is_constructible<value_type, Args...> && detail::is_constructible<error_type, Args...>, //
    disable_inplace_value_error_constructor, //
    std::conditional_t< //
    detail::is_constructible<value_type, Args...>, //
    value_type, //
    std::conditional_t< //
    detail::is_constructible<error_type, Args...>, //
    error_type, //
    disable_inplace_value_error_constructor>>>;
    template <class... Args>
    static constexpr bool enable_inplace_value_error_constructor =
    implicit_constructors_enabled //
    && !std::is_same<choose_inplace_value_error_constructor<Args...>, disable_inplace_value_error_constructor>::value;
  };
  template <class T, class U> constexpr inline const U &extract_value_from_success(const success_type<U> &v) { return v.value(); }
  template <class T, class U> constexpr inline U &&extract_value_from_success(success_type<U> &&v) { return static_cast<success_type<U> &&>(v).value(); }
  template <class T> constexpr inline T extract_value_from_success(const success_type<void> & /*unused*/) { return T{}; }
  template <class T, class U, class V> constexpr inline const U &extract_error_from_failure(const failure_type<U, V> &v) { return v.error(); }
  template <class T, class U, class V> constexpr inline U &&extract_error_from_failure(failure_type<U, V> &&v)
  {
    return static_cast<failure_type<U, V> &&>(v).error();
  }
  template <class T, class V> constexpr inline T extract_error_from_failure(const failure_type<void, V> & /*unused*/) { return T{}; }
  template <class T> struct is_basic_result
  {
    static constexpr bool value = false;
  };
  template <class R, class S, class T> struct is_basic_result<basic_result<R, S, T>>
  {
    static constexpr bool value = true;
  };
} // namespace detail
/*! AWAITING HUGO JSON CONVERSION TOOL
type alias template <class T> is_basic_result. Potential doc page: `is_basic_result<T>`
*/
template <class T> using is_basic_result = detail::is_basic_result<std::decay_t<T>>;
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class T> static constexpr bool is_basic_result_v = detail::is_basic_result<std::decay_t<T>>::value;
namespace concepts
{
#if defined(__cpp_concepts)
  /* The `basic_result` concept.
  \requires That `U` matches a `basic_result`.
  */
  template <class U>
  concept OUTCOME_GCC6_CONCEPT_BOOL basic_result =
  OUTCOME_V2_NAMESPACE::is_basic_result<U>::value ||
  (requires(U v) { OUTCOME_V2_NAMESPACE::basic_result<typename U::value_type, typename U::error_type, typename U::no_value_policy_type>(v); } && //
   detail::convertible<U, OUTCOME_V2_NAMESPACE::basic_result<typename U::value_type, typename U::error_type, typename U::no_value_policy_type>> && //
   detail::base_of<OUTCOME_V2_NAMESPACE::basic_result<typename U::value_type, typename U::error_type, typename U::no_value_policy_type>, U>);
#else
  namespace detail
  {
    inline no_match match_basic_result(...);
    template <class R, class S, class NVP, class T, //
              typename = typename T::value_type, //
              typename = typename T::error_type, //
              typename = typename T::no_value_policy_type, //
              typename std::enable_if_t<std::is_convertible<T, OUTCOME_V2_NAMESPACE::basic_result<R, S, NVP>>::value && //
                                        std::is_base_of<OUTCOME_V2_NAMESPACE::basic_result<R, S, NVP>, T>::value,
                                        bool> = true>
    inline OUTCOME_V2_NAMESPACE::basic_result<R, S, NVP> match_basic_result(OUTCOME_V2_NAMESPACE::basic_result<R, S, NVP> &&, T &&);
    template <class U>
    static constexpr bool basic_result = OUTCOME_V2_NAMESPACE::is_basic_result<U>::value ||
                                         !std::is_same<no_match, decltype(match_basic_result(std::declval<OUTCOME_V2_NAMESPACE::detail::devoid<U>>(),
                                                                                             std::declval<OUTCOME_V2_NAMESPACE::detail::devoid<U>>()))>::value;
  } // namespace detail
  /* The `basic_result` concept.
  \requires That `U` matches a `basic_result`.
  */
  template <class U> static constexpr bool basic_result = detail::basic_result<U>;
#endif
} // namespace concepts
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
namespace hooks
{
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class R, class S, class NoValuePolicy> constexpr inline uint16_t spare_storage(const detail::basic_result_storage<R, S, NoValuePolicy> *r) noexcept
  {
    return r->_state._status.spare_storage_value;
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class R, class S, class NoValuePolicy>
  constexpr inline void set_spare_storage(detail::basic_result_storage<R, S, NoValuePolicy> *r, uint16_t v) noexcept
  {
    r->_state._status.spare_storage_value = v;
  }
} // namespace hooks
/*! AWAITING HUGO JSON CONVERSION TOOL
type definition template <class R, class S, class NoValuePolicy> basic_result. Potential doc page: `basic_result<T, E, NoValuePolicy>`
*/
template <class R, class S, class NoValuePolicy> //
class OUTCOME_NODISCARD basic_result : public detail::basic_result_final<R, S, NoValuePolicy>
{
  static_assert(trait::type_can_be_used_in_basic_result<R>, "The type R cannot be used in a basic_result");
  static_assert(trait::type_can_be_used_in_basic_result<S>, "The type S cannot be used in a basic_result");
  using base = detail::basic_result_final<R, S, NoValuePolicy>;
  struct implicit_constructors_disabled_tag
  {
  };
  struct value_converting_constructor_tag
  {
  };
  struct error_converting_constructor_tag
  {
  };
  struct error_condition_converting_constructor_tag
  {
  };
  struct explicit_valueornone_converting_constructor_tag
  {
  };
  struct explicit_valueorerror_converting_constructor_tag
  {
  };
  struct explicit_compatible_copy_conversion_tag
  {
  };
  struct explicit_compatible_move_conversion_tag
  {
  };
  struct explicit_make_error_code_compatible_copy_conversion_tag
  {
  };
  struct explicit_make_error_code_compatible_move_conversion_tag
  {
  };
  struct explicit_make_exception_ptr_compatible_copy_conversion_tag
  {
  };
  struct explicit_make_exception_ptr_compatible_move_conversion_tag
  {
  };
public:
  using value_type = R;
  using error_type = S;
  using no_value_policy_type = NoValuePolicy;
  using value_type_if_enabled = typename base::_value_type;
  using error_type_if_enabled = typename base::_error_type;
  template <class T, class U = S, class V = NoValuePolicy> using rebind = basic_result<T, U, V>;
protected:
  // Requirement predicates for result.
  struct predicate
  {
    using base = detail::result_predicates<value_type, error_type>;
    // Predicate for any constructors to be available at all
    static constexpr bool constructors_enabled = !std::is_same<std::decay_t<value_type>, std::decay_t<error_type>>::value;
    // Predicate for implicit constructors to be available at all
    static constexpr bool implicit_constructors_enabled = constructors_enabled && base::implicit_constructors_enabled;
    // Predicate for the value converting constructor to be available.
    template <class T>
    static constexpr bool enable_value_converting_constructor = //
    constructors_enabled //
    && !std::is_same<std::decay_t<T>, basic_result>::value // not my type
    && base::template enable_value_converting_constructor<T>;
    // Predicate for the error converting constructor to be available.
    template <class T>
    static constexpr bool enable_error_converting_constructor = //
    constructors_enabled //
    && !std::is_same<std::decay_t<T>, basic_result>::value // not my type
    && base::template enable_error_converting_constructor<T>;
    // Predicate for the error condition converting constructor to be available.
    template <class ErrorCondEnum>
    static constexpr bool enable_error_condition_converting_constructor = //
    constructors_enabled //
    && !std::is_same<std::decay_t<ErrorCondEnum>, basic_result>::value // not my type
    && base::template enable_error_condition_converting_constructor<ErrorCondEnum>;
    // Predicate for the converting constructor from a compatible input to be available.
    template <class T, class U, class V>
    static constexpr bool enable_compatible_conversion = //
    constructors_enabled //
    && !std::is_same<basic_result<T, U, V>, basic_result>::value // not my type
    && base::template enable_compatible_conversion<T, U, V>;
    // Predicate for the converting constructor from a make_error_code() of the input to be available.
    template <class T, class U, class V>
    static constexpr bool enable_make_error_code_compatible_conversion = //
    constructors_enabled //
    && !std::is_same<basic_result<T, U, V>, basic_result>::value // not my type
    && base::template enable_make_error_code_compatible_conversion<T, U, V>;
    // Predicate for the converting constructor from a make_exception_ptr() of the input to be available.
    template <class T, class U, class V>
    static constexpr bool enable_make_exception_ptr_compatible_conversion = //
    constructors_enabled //
    && !std::is_same<basic_result<T, U, V>, basic_result>::value // not my type
    && base::template enable_make_exception_ptr_compatible_conversion<T, U, V>;
    // Predicate for the inplace construction of value to be available.
    template <class... Args>
    static constexpr bool enable_inplace_value_constructor = //
    constructors_enabled //
    && (std::is_void<value_type>::value //
        || detail::is_constructible<value_type, Args...>);
    // Predicate for the inplace construction of error to be available.
    template <class... Args>
    static constexpr bool enable_inplace_error_constructor = //
    constructors_enabled //
    && (std::is_void<error_type>::value //
        || detail::is_constructible<error_type, Args...>);
    // Predicate for the implicit converting inplace constructor to be available.
    template <class... Args>
    static constexpr bool enable_inplace_value_error_constructor = //
    constructors_enabled //
    &&base::template enable_inplace_value_error_constructor<Args...>;
    template <class... Args> using choose_inplace_value_error_constructor = typename base::template choose_inplace_value_error_constructor<Args...>;
  };
public:
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  basic_result() = delete;
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  basic_result(basic_result && /*unused*/) = default; // NOLINT
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  basic_result(const basic_result & /*unused*/) = default;
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  basic_result &operator=(basic_result && /*unused*/) = default; // NOLINT
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  basic_result &operator=(const basic_result & /*unused*/) = default;
  ~basic_result() = default;
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class Arg, class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!predicate::constructors_enabled && (sizeof...(Args) >= 0)))
  basic_result(Arg && /*unused*/, Args &&... /*unused*/) = delete; // NOLINT basic_result<T, T> is NOT SUPPORTED, see docs!
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED((predicate::constructors_enabled && !predicate::implicit_constructors_enabled //
                                   && (detail::is_implicitly_constructible<value_type, T> || detail::is_implicitly_constructible<error_type, T>) )))
  basic_result(T && /*unused*/, implicit_constructors_disabled_tag /*unused*/ = implicit_constructors_disabled_tag()) =
  delete; // NOLINT Implicit constructors disabled, use explicit in_place_type<T>, success() or failure(). see docs!
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_value_converting_constructor<T>))
  constexpr basic_result(T &&t, value_converting_constructor_tag /*unused*/ = value_converting_constructor_tag()) noexcept(
  detail::is_nothrow_constructible<value_type, T>) // NOLINT
      : base{in_place_type<typename base::value_type>, static_cast<T &&>(t)}
  {
    no_value_policy_type::on_result_construction(this, static_cast<T &&>(t));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_error_converting_constructor<T>))
  constexpr basic_result(T &&t, error_converting_constructor_tag /*unused*/ = error_converting_constructor_tag()) noexcept(
  detail::is_nothrow_constructible<error_type, T>) // NOLINT
      : base{in_place_type<typename base::error_type>, static_cast<T &&>(t)}
  {
    no_value_policy_type::on_result_construction(this, static_cast<T &&>(t));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class ErrorCondEnum)
  OUTCOME_TREQUIRES(OUTCOME_TEXPR(error_type(make_error_code(ErrorCondEnum()))), //
                    OUTCOME_TPRED(predicate::template enable_error_condition_converting_constructor<ErrorCondEnum>))
  constexpr basic_result(ErrorCondEnum &&t, error_condition_converting_constructor_tag /*unused*/ = error_condition_converting_constructor_tag()) noexcept(
  noexcept(error_type(make_error_code(static_cast<ErrorCondEnum &&>(t))))) // NOLINT
      : base{in_place_type<typename base::error_type>, make_error_code(t)}
  {
    no_value_policy_type::on_result_construction(this, static_cast<ErrorCondEnum &&>(t));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(convert::value_or_error<basic_result, std::decay_t<T>>::enable_result_inputs || !concepts::basic_result<T>), //
                    OUTCOME_TEXPR(convert::value_or_error<basic_result, std::decay_t<T>>{}(std::declval<T>())))
  constexpr explicit basic_result(T &&o,
                                  explicit_valueorerror_converting_constructor_tag /*unused*/ = explicit_valueorerror_converting_constructor_tag()) // NOLINT
      : basic_result{convert::value_or_error<basic_result, std::decay_t<T>>{}(static_cast<T &&>(o))}
  {
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_compatible_conversion<T, U, V>))
  constexpr explicit basic_result(
  const basic_result<T, U, V> &o,
  explicit_compatible_copy_conversion_tag /*unused*/ =
  explicit_compatible_copy_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T> &&detail::is_nothrow_constructible<error_type, U>)
      : base{typename base::compatible_conversion_tag(), o}
  {
    no_value_policy_type::on_result_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_compatible_conversion<T, U, V>))
  constexpr explicit basic_result(
  basic_result<T, U, V> &&o,
  explicit_compatible_move_conversion_tag /*unused*/ =
  explicit_compatible_move_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T> &&detail::is_nothrow_constructible<error_type, U>)
      : base{typename base::compatible_conversion_tag(), static_cast<basic_result<T, U, V> &&>(o)}
  {
    no_value_policy_type::on_result_move_construction(this, static_cast<basic_result<T, U, V> &&>(o));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_make_error_code_compatible_conversion<T, U, V>))
  constexpr explicit basic_result(const basic_result<T, U, V> &o,
                                  explicit_make_error_code_compatible_copy_conversion_tag /*unused*/ =
                                  explicit_make_error_code_compatible_copy_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T>
                                                                                                      &&noexcept(make_error_code(std::declval<U>())))
      : base{typename base::make_error_code_compatible_conversion_tag(), o}
  {
    no_value_policy_type::on_result_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_make_error_code_compatible_conversion<T, U, V>))
  constexpr explicit basic_result(basic_result<T, U, V> &&o,
                                  explicit_make_error_code_compatible_move_conversion_tag /*unused*/ =
                                  explicit_make_error_code_compatible_move_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T>
                                                                                                      &&noexcept(make_error_code(std::declval<U>())))
      : base{typename base::make_error_code_compatible_conversion_tag(), static_cast<basic_result<T, U, V> &&>(o)}
  {
    no_value_policy_type::on_result_move_construction(this, static_cast<basic_result<T, U, V> &&>(o));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_make_exception_ptr_compatible_conversion<T, U, V>))
  constexpr explicit basic_result(const basic_result<T, U, V> &o,
                                  explicit_make_exception_ptr_compatible_copy_conversion_tag /*unused*/ =
                                  explicit_make_exception_ptr_compatible_copy_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T>
                                                                                                         &&noexcept(make_exception_ptr(std::declval<U>())))
      : base{typename base::make_exception_ptr_compatible_conversion_tag(), o}
  {
    no_value_policy_type::on_result_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_make_exception_ptr_compatible_conversion<T, U, V>))
  constexpr explicit basic_result(basic_result<T, U, V> &&o,
                                  explicit_make_exception_ptr_compatible_move_conversion_tag /*unused*/ =
                                  explicit_make_exception_ptr_compatible_move_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T>
                                                                                                         &&noexcept(make_exception_ptr(std::declval<U>())))
      : base{typename base::make_exception_ptr_compatible_conversion_tag(), static_cast<basic_result<T, U, V> &&>(o)}
  {
    no_value_policy_type::on_result_move_construction(this, static_cast<basic_result<T, U, V> &&>(o));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_inplace_value_constructor<Args...>))
  constexpr explicit basic_result(in_place_type_t<value_type_if_enabled> _, Args &&... args) noexcept(detail::is_nothrow_constructible<value_type, Args...>)
      : base{_, static_cast<Args &&>(args)...}
  {
    no_value_policy_type::on_result_in_place_construction(this, in_place_type<value_type>, static_cast<Args &&>(args)...);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class U, class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_inplace_value_constructor<std::initializer_list<U>, Args...>))
  constexpr explicit basic_result(in_place_type_t<value_type_if_enabled> _, std::initializer_list<U> il,
                                  Args &&... args) noexcept(detail::is_nothrow_constructible<value_type, std::initializer_list<U>, Args...>)
      : base{_, il, static_cast<Args &&>(args)...}
  {
    no_value_policy_type::on_result_in_place_construction(this, in_place_type<value_type>, il, static_cast<Args &&>(args)...);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_inplace_error_constructor<Args...>))
  constexpr explicit basic_result(in_place_type_t<error_type_if_enabled> _, Args &&... args) noexcept(detail::is_nothrow_constructible<error_type, Args...>)
      : base{_, static_cast<Args &&>(args)...}
  {
    no_value_policy_type::on_result_in_place_construction(this, in_place_type<error_type>, static_cast<Args &&>(args)...);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class U, class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_inplace_error_constructor<std::initializer_list<U>, Args...>))
  constexpr explicit basic_result(in_place_type_t<error_type_if_enabled> _, std::initializer_list<U> il,
                                  Args &&... args) noexcept(detail::is_nothrow_constructible<error_type, std::initializer_list<U>, Args...>)
      : base{_, il, static_cast<Args &&>(args)...}
  {
    no_value_policy_type::on_result_in_place_construction(this, in_place_type<error_type>, il, static_cast<Args &&>(args)...);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class A1, class A2, class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_inplace_value_error_constructor<A1, A2, Args...>))
  constexpr basic_result(A1 &&a1, A2 &&a2, Args &&... args) noexcept(noexcept(
  typename predicate::template choose_inplace_value_error_constructor<A1, A2, Args...>(std::declval<A1>(), std::declval<A2>(), std::declval<Args>()...)))
      : basic_result(in_place_type<typename predicate::template choose_inplace_value_error_constructor<A1, A2, Args...>>, static_cast<A1 &&>(a1),
                     static_cast<A2 &&>(a2), static_cast<Args &&>(args)...)
  {
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  constexpr basic_result(const success_type<void> &o) noexcept(std::is_nothrow_default_constructible<value_type>::value) // NOLINT
      : base{in_place_type<value_type_if_enabled>}
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_result_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_compatible_conversion<T, void, void>))
  constexpr basic_result(const success_type<T> &o) noexcept(detail::is_nothrow_constructible<value_type, T>) // NOLINT
      : base{in_place_type<value_type_if_enabled>, detail::extract_value_from_success<value_type>(o)}
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_result_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_compatible_conversion<T, void, void>))
  constexpr basic_result(success_type<T> &&o) noexcept(detail::is_nothrow_constructible<value_type, T>) // NOLINT
      : base{in_place_type<value_type_if_enabled>, detail::extract_value_from_success<value_type>(static_cast<success_type<T> &&>(o))}
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_result_move_construction(this, static_cast<success_type<T> &&>(o));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_compatible_conversion<void, T, void>))
  constexpr basic_result(const failure_type<T> &o, explicit_compatible_copy_conversion_tag /*unused*/ = explicit_compatible_copy_conversion_tag()) noexcept(
  detail::is_nothrow_constructible<error_type, T>) // NOLINT
      : base{in_place_type<error_type_if_enabled>, detail::extract_error_from_failure<error_type>(o)}
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_result_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_compatible_conversion<void, T, void>))
  constexpr basic_result(failure_type<T> &&o, explicit_compatible_move_conversion_tag /*unused*/ = explicit_compatible_move_conversion_tag()) noexcept(
  detail::is_nothrow_constructible<error_type, T>) // NOLINT
      : base{in_place_type<error_type_if_enabled>, detail::extract_error_from_failure<error_type>(static_cast<failure_type<T> &&>(o))}
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_result_move_construction(this, static_cast<failure_type<T> &&>(o));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_make_error_code_compatible_conversion<void, T, void>))
  constexpr basic_result(const failure_type<T> &o,
                         explicit_make_error_code_compatible_copy_conversion_tag /*unused*/ =
                         explicit_make_error_code_compatible_copy_conversion_tag()) noexcept(noexcept(make_error_code(std::declval<T>()))) // NOLINT
      : base{in_place_type<error_type_if_enabled>, make_error_code(detail::extract_error_from_failure<error_type>(o))}
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_result_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_make_error_code_compatible_conversion<void, T, void>))
  constexpr basic_result(failure_type<T> &&o,
                         explicit_make_error_code_compatible_move_conversion_tag /*unused*/ =
                         explicit_make_error_code_compatible_move_conversion_tag()) noexcept(noexcept(make_error_code(std::declval<T>()))) // NOLINT
      : base{in_place_type<error_type_if_enabled>, make_error_code(detail::extract_error_from_failure<error_type>(static_cast<failure_type<T> &&>(o)))}
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_result_move_construction(this, static_cast<failure_type<T> &&>(o));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_make_exception_ptr_compatible_conversion<void, T, void>))
  constexpr basic_result(const failure_type<T> &o,
                         explicit_make_exception_ptr_compatible_copy_conversion_tag /*unused*/ =
                         explicit_make_exception_ptr_compatible_copy_conversion_tag()) noexcept(noexcept(make_exception_ptr(std::declval<T>()))) // NOLINT
      : base{in_place_type<error_type_if_enabled>, make_exception_ptr(detail::extract_error_from_failure<error_type>(o))}
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_result_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_make_exception_ptr_compatible_conversion<void, T, void>))
  constexpr basic_result(failure_type<T> &&o,
                         explicit_make_exception_ptr_compatible_move_conversion_tag /*unused*/ =
                         explicit_make_exception_ptr_compatible_move_conversion_tag()) noexcept(noexcept(make_exception_ptr(std::declval<T>()))) // NOLINT
      : base{in_place_type<error_type_if_enabled>, make_exception_ptr(detail::extract_error_from_failure<error_type>(static_cast<failure_type<T> &&>(o)))}
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_result_move_construction(this, static_cast<failure_type<T> &&>(o));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  constexpr void swap(basic_result &o) noexcept((std::is_void<value_type>::value || detail::is_nothrow_swappable<value_type>::value) //
                                                && (std::is_void<error_type>::value || detail::is_nothrow_swappable<error_type>::value))
  {
    this->_state.swap(o._state);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  auto as_failure() const & { return failure(this->assume_error(), hooks::spare_storage(this)); }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  auto as_failure() &&
  {
    this->_state._status.set_have_moved_from(true);
    return failure(static_cast<basic_result &&>(*this).assume_error(), hooks::spare_storage(this));
  }
#ifdef __APPLE__
  failure_type<error_type> _xcode_workaround_as_failure() &&;
#endif
};
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class R, class S, class P> inline void swap(basic_result<R, S, P> &a, basic_result<R, S, P> &b) noexcept(noexcept(a.swap(b)))
{
  a.swap(b);
}
#if !defined(NDEBUG)
// Check is trivial in all ways except default constructibility
// static_assert(std::is_trivial<basic_result<int, long, policy::all_narrow>>::value, "result<int> is not trivial!");
// static_assert(std::is_trivially_default_constructible<basic_result<int, long, policy::all_narrow>>::value, "result<int> is not trivially default
// constructible!");
static_assert(std::is_trivially_copyable<basic_result<int, long, policy::all_narrow>>::value, "result<int> is not trivially copyable!");
static_assert(std::is_trivially_assignable<basic_result<int, long, policy::all_narrow>, basic_result<int, long, policy::all_narrow>>::value,
              "result<int> is not trivially assignable!");
static_assert(std::is_trivially_destructible<basic_result<int, long, policy::all_narrow>>::value, "result<int> is not trivially destructible!");
static_assert(std::is_trivially_copy_constructible<basic_result<int, long, policy::all_narrow>>::value, "result<int> is not trivially copy constructible!");
static_assert(std::is_trivially_move_constructible<basic_result<int, long, policy::all_narrow>>::value, "result<int> is not trivially move constructible!");
static_assert(std::is_trivially_copy_assignable<basic_result<int, long, policy::all_narrow>>::value, "result<int> is not trivially copy assignable!");
static_assert(std::is_trivially_move_assignable<basic_result<int, long, policy::all_narrow>>::value, "result<int> is not trivially move assignable!");
// Also check is standard layout
static_assert(std::is_standard_layout<basic_result<int, long, policy::all_narrow>>::value, "result<int> is not a standard layout type!");
#endif
OUTCOME_V2_NAMESPACE_END
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#endif
/* Traits for Outcome
(C) 2018-2019 Niall Douglas <http://www.nedproductions.biz/> (6 commits)
File Created: March 2018


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_TRAIT_STD_ERROR_CODE_HPP
#define OUTCOME_TRAIT_STD_ERROR_CODE_HPP
#include <system_error>
OUTCOME_V2_NAMESPACE_BEGIN
namespace detail
{
  // Customise _set_error_is_errno
  template <class State> constexpr inline void _set_error_is_errno(State &state, const std::error_code &error)
  {
    if(error.category() == std::generic_category()
#ifndef _WIN32
       || error.category() == std::system_category()
#endif
    )
    {
      state._status.set_have_error_is_errno(true);
    }
  }
  template <class State> constexpr inline void _set_error_is_errno(State &state, const std::error_condition &error)
  {
    if(error.category() == std::generic_category()
#ifndef _WIN32
       || error.category() == std::system_category()
#endif
    )
    {
      state._status.set_have_error_is_errno(true);
    }
  }
  template <class State> constexpr inline void _set_error_is_errno(State &state, const std::errc & /*unused*/) {
      state._status.set_have_error_is_errno(true);
   }
} // namespace detail
namespace policy
{
  namespace detail
  {
    /* Pass through `make_error_code` function for `std::error_code`.
     */
    inline std::error_code make_error_code(std::error_code v) { return v; }
    // Try ADL, if not use fall backs above
    template <class T> constexpr inline decltype(auto) error_code(T &&v) { return make_error_code(std::forward<T>(v)); }
    struct std_enum_overload_tag
    {
    };
  } // namespace detail
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class T> constexpr inline decltype(auto) error_code(T &&v) { return detail::error_code(std::forward<T>(v)); }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  // inline void outcome_throw_as_system_error_with_payload(...) = delete;  // To use the error_code_throw_as_system_error policy with a custom Error type, you must define a outcome_throw_as_system_error_with_payload() free function to say how to handle the payload
  inline void outcome_throw_as_system_error_with_payload(const std::error_code &error) { OUTCOME_THROW_EXCEPTION(std::system_error(error)); } // NOLINT
  OUTCOME_TEMPLATE(class Error)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(std::is_error_code_enum<std::decay_t<Error>>::value || std::is_error_condition_enum<std::decay_t<Error>>::value))
  inline void outcome_throw_as_system_error_with_payload(Error &&error, detail::std_enum_overload_tag /*unused*/ = detail::std_enum_overload_tag()) { OUTCOME_THROW_EXCEPTION(std::system_error(make_error_code(error))); } // NOLINT
} // namespace policy
namespace trait
{
  namespace detail
  {
    template <> struct _is_error_code_available<std::error_code>
    {
      // Shortcut this for lower build impact
      static constexpr bool value = true;
      using type = std::error_code;
    };
  } // namespace detail
  // std::error_code is an error type
  template <> struct is_error_type<std::error_code>
  {
    static constexpr bool value = true;
  };
  // For std::error_code, std::is_error_condition_enum<> is the trait we want.
  template <class Enum> struct is_error_type_enum<std::error_code, Enum>
  {
    static constexpr bool value = std::is_error_condition_enum<Enum>::value;
  };
} // namespace trait
OUTCOME_V2_NAMESPACE_END
#endif
/* Traits for Outcome
(C) 2018-2019 Niall Douglas <http://www.nedproductions.biz/> (3 commits)
File Created: March 2018


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_TRAIT_STD_EXCEPTION_HPP
#define OUTCOME_TRAIT_STD_EXCEPTION_HPP
#include <exception>
OUTCOME_V2_NAMESPACE_BEGIN
namespace policy
{
  namespace detail
  {
    /* Pass through `make_exception_ptr` function for `std::exception_ptr`.
    */
    inline std::exception_ptr make_exception_ptr(std::exception_ptr v) { return v; }
    // Try ADL, if not use fall backs above
    template <class T> constexpr inline decltype(auto) exception_ptr(T &&v) { return make_exception_ptr(std::forward<T>(v)); }
  } // namespace detail
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class T> constexpr inline decltype(auto) exception_ptr(T &&v) { return detail::exception_ptr(std::forward<T>(v)); }
  namespace detail
  {
    template <bool has_error_payload> struct _rethrow_exception
    {
      template <class Exception> explicit _rethrow_exception(Exception && /*unused*/) // NOLINT
      {
      }
    };
    template <> struct _rethrow_exception<true>
    {
      template <class Exception> explicit _rethrow_exception(Exception &&excpt) // NOLINT
      {
        // ADL
        rethrow_exception(policy::exception_ptr(std::forward<Exception>(excpt)));
      }
    };
  } // namespace detail
} // namespace policy
namespace trait
{
  namespace detail
  {
    // Shortcut this for lower build impact
    template <> struct _is_exception_ptr_available<std::exception_ptr>
    {
      static constexpr bool value = true;
      using type = std::exception_ptr;
    };
  } // namespace detail
  // std::exception_ptr is an error type
  template <> struct is_error_type<std::exception_ptr>
  {
    static constexpr bool value = true;
  };
} // namespace trait
OUTCOME_V2_NAMESPACE_END
#endif
/* Policies for result and outcome
(C) 2018-2019 Niall Douglas <http://www.nedproductions.biz/> (4 commits)
File Created: Sep 2018


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_POLICY_FAIL_TO_COMPILE_OBSERVERS_HPP
#define OUTCOME_POLICY_FAIL_TO_COMPILE_OBSERVERS_HPP
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
#define OUTCOME_FAIL_TO_COMPILE_OBSERVERS_MESSAGE "Attempt to wide observe value, error or " "exception for a basic_result/basic_outcome given an EC or EP type which is not void, and for whom " "trait::is_error_code_available<EC>, trait::is_exception_ptr_available<EC>, and trait::is_exception_ptr_available<EP> " "are all false. Please specify a NoValuePolicy to tell basic_result/basic_outcome what to do, or else use " "a more specific convenience type alias such as unchecked<T, E> to indicate you want the wide " "observers to be narrow, or checked<T, E> to indicate you always want an exception throw etc."
namespace policy
{
  struct fail_to_compile_observers : base
  {
    template <class Impl> static constexpr void wide_value_check(Impl && /* unused */) { static_assert(!std::is_same<Impl, Impl>::value, "Attempt to wide observe value, error or " "exception for a basic_result/basic_outcome given an EC or EP type which is not void, and for whom " "trait::is_error_code_available<EC>, trait::is_exception_ptr_available<EC>, and trait::is_exception_ptr_available<EP> " "are all false. Please specify a NoValuePolicy to tell basic_result/basic_outcome what to do, or else use " "a more specific convenience type alias such as unchecked<T, E> to indicate you want the wide " "observers to be narrow, or checked<T, E> to indicate you always want an exception throw etc."); }
    template <class Impl> static constexpr void wide_error_check(Impl && /* unused */) { static_assert(!std::is_same<Impl, Impl>::value, "Attempt to wide observe value, error or " "exception for a basic_result/basic_outcome given an EC or EP type which is not void, and for whom " "trait::is_error_code_available<EC>, trait::is_exception_ptr_available<EC>, and trait::is_exception_ptr_available<EP> " "are all false. Please specify a NoValuePolicy to tell basic_result/basic_outcome what to do, or else use " "a more specific convenience type alias such as unchecked<T, E> to indicate you want the wide " "observers to be narrow, or checked<T, E> to indicate you always want an exception throw etc."); }
    template <class Impl> static constexpr void wide_exception_check(Impl && /* unused */) { static_assert(!std::is_same<Impl, Impl>::value, "Attempt to wide observe value, error or " "exception for a basic_result/basic_outcome given an EC or EP type which is not void, and for whom " "trait::is_error_code_available<EC>, trait::is_exception_ptr_available<EC>, and trait::is_exception_ptr_available<EP> " "are all false. Please specify a NoValuePolicy to tell basic_result/basic_outcome what to do, or else use " "a more specific convenience type alias such as unchecked<T, E> to indicate you want the wide " "observers to be narrow, or checked<T, E> to indicate you always want an exception throw etc."); }
  };
} // namespace policy
#undef OUTCOME_FAIL_TO_COMPILE_OBSERVERS_MESSAGE
OUTCOME_V2_NAMESPACE_END
#endif
/* Policies for result and outcome
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (8 commits)
File Created: Oct 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_POLICY_RESULT_ERROR_CODE_THROW_AS_SYSTEM_ERROR_HPP
#define OUTCOME_POLICY_RESULT_ERROR_CODE_THROW_AS_SYSTEM_ERROR_HPP
/* Exception types throwable
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (9 commits)
File Created: Oct 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_BAD_ACCESS_HPP
#define OUTCOME_BAD_ACCESS_HPP
#include <stdexcept>
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
/*! AWAITING HUGO JSON CONVERSION TOOL
type definition  bad_result_access. Potential doc page: `bad_result_access`
*/
class OUTCOME_SYMBOL_VISIBLE bad_result_access : public std::logic_error
{
public:
  explicit bad_result_access(const char *what)
      : std::logic_error(what)
  {
  }
};
/*! AWAITING HUGO JSON CONVERSION TOOL
type definition template <class S> bad_result_access_with. Potential doc page: `bad_result_access_with<EC>`
*/
template <class S> class OUTCOME_SYMBOL_VISIBLE bad_result_access_with : public bad_result_access
{
  S _error;
public:
  explicit bad_result_access_with(S v)
      : bad_result_access("no value")
      , _error(std::move(v))
  {
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  const S &error() const & { return _error; }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  S &error() & { return _error; }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  const S &&error() const && { return _error; }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  S &&error() && { return _error; }
};
/*! AWAITING HUGO JSON CONVERSION TOOL
type definition  bad_outcome_access. Potential doc page: `bad_outcome_access`
*/
class OUTCOME_SYMBOL_VISIBLE bad_outcome_access : public std::logic_error
{
public:
  explicit bad_outcome_access(const char *what)
      : std::logic_error(what)
  {
  }
};
OUTCOME_V2_NAMESPACE_END
#endif
#include <system_error>
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace policy
{
  template <class T, class EC, class E> struct error_code_throw_as_system_error;
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class T, class EC> struct error_code_throw_as_system_error<T, EC, void> : base
  {
    template <class Impl> static constexpr void wide_value_check(Impl &&self)
    {
      if(!base::_has_value(std::forward<Impl>(self)))
      {
        if(base::_has_error(std::forward<Impl>(self)))
        {
          // ADL discovered
          outcome_throw_as_system_error_with_payload(base::_error(std::forward<Impl>(self)));
        }
        OUTCOME_THROW_EXCEPTION(bad_result_access("no value")); // NOLINT
      }
    }
    template <class Impl> static constexpr void wide_error_check(Impl &&self)
    {
      if(!base::_has_error(std::forward<Impl>(self)))
      {
        OUTCOME_THROW_EXCEPTION(bad_result_access("no error")); // NOLINT
      }
    }
  };
} // namespace policy
OUTCOME_V2_NAMESPACE_END
#endif
/* Policies for result and outcome
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (6 commits)
File Created: Oct 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_POLICY_RESULT_EXCEPTION_PTR_RETHROW_HPP
#define OUTCOME_POLICY_RESULT_EXCEPTION_PTR_RETHROW_HPP
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace policy
{
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class T, class EC, class E> struct exception_ptr_rethrow;
  template <class T, class EC> struct exception_ptr_rethrow<T, EC, void> : base
  {
    template <class Impl> static constexpr void wide_value_check(Impl &&self)
    {
      if(!base::_has_value(std::forward<Impl>(self)))
      {
        if(base::_has_error(std::forward<Impl>(self)))
        {
          // ADL
          rethrow_exception(policy::exception_ptr(base::_error(std::forward<Impl>(self))));
        }
        OUTCOME_THROW_EXCEPTION(bad_result_access("no value")); // NOLINT
      }
    }
    template <class Impl> static constexpr void wide_error_check(Impl &&self)
    {
      if(!base::_has_error(std::forward<Impl>(self)))
      {
        OUTCOME_THROW_EXCEPTION(bad_result_access("no error")); // NOLINT
      }
    }
  };
} // namespace policy
OUTCOME_V2_NAMESPACE_END
#endif
/* Policies for result and outcome
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (13 commits)
File Created: Oct 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_POLICY_THROW_BAD_RESULT_ACCESS_HPP
#define OUTCOME_POLICY_THROW_BAD_RESULT_ACCESS_HPP
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace policy
{
  /*! AWAITING HUGO JSON CONVERSION TOOL
type definition  throw_bad_result_access. Potential doc page: NOT FOUND
*/
  template <class EC, class EP> struct throw_bad_result_access : base
  {
    template <class Impl> static constexpr void wide_value_check(Impl &&self)
    {
      if(!base::_has_value(std::forward<Impl>(self)))
      {
        OUTCOME_THROW_EXCEPTION(bad_outcome_access("no value")); // NOLINT
      }
    }
    template <class Impl> static constexpr void wide_error_check(Impl &&self)
    {
      if(!base::_has_error(std::forward<Impl>(self)))
      {
        OUTCOME_THROW_EXCEPTION(bad_outcome_access("no error")); // NOLINT
      }
    }
    template <class Impl> static constexpr void wide_exception_check(Impl &&self)
    {
      if(!base::_has_exception(std::forward<Impl>(self)))
      {
        OUTCOME_THROW_EXCEPTION(bad_outcome_access("no exception")); // NOLINT
      }
    }
  };
  template <class EC> struct throw_bad_result_access<EC, void> : base
  {
    template <class Impl> static constexpr void wide_value_check(Impl &&self)
    {
      if(!base::_has_value(std::forward<Impl>(self)))
      {
        if(base::_has_error(std::forward<Impl>(self)))
        {
          OUTCOME_THROW_EXCEPTION(bad_result_access_with<EC>(base::_error(std::forward<Impl>(self))));
        }
        OUTCOME_THROW_EXCEPTION(bad_result_access("no value")); // NOLINT
      }
    }
    template <class Impl> static constexpr void wide_error_check(Impl &&self)
    {
      if(!base::_has_error(std::forward<Impl>(self)))
      {
        OUTCOME_THROW_EXCEPTION(bad_result_access("no error")); // NOLINT
      }
    }
  };
} // namespace policy
OUTCOME_V2_NAMESPACE_END
#endif
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
namespace policy
{
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class T, class EC, class E>
  using default_policy = std::conditional_t< //
  std::is_void<EC>::value && std::is_void<E>::value,
  terminate, //
  std::conditional_t< //
  trait::is_error_code_available<EC>::value, error_code_throw_as_system_error<T, EC, E>, //
  std::conditional_t< //
  trait::is_exception_ptr_available<EC>::value || trait::is_exception_ptr_available<E>::value, exception_ptr_rethrow<T, EC, E>, //
  fail_to_compile_observers //
  >>>;
} // namespace policy
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class R, class S = std::error_code, class NoValuePolicy = policy::default_policy<R, S, void>> //
using std_result = basic_result<R, S, NoValuePolicy>;
/*! AWAITING HUGO JSON CONVERSION TOOL
type alias template <class R, class S = std::error_code> std_unchecked. Potential doc page: `std_unchecked<T, E = std::error_code>`
*/
template <class R, class S = std::error_code> using std_unchecked = std_result<R, S, policy::all_narrow>;
/*! AWAITING HUGO JSON CONVERSION TOOL
type alias template <class R, class S = std::error_code> std_checked. Potential doc page: `std_checked<T, E = std::error_code>`
*/
template <class R, class S = std::error_code> using std_checked = std_result<R, S, policy::throw_bad_result_access<S, void>>;
OUTCOME_V2_NAMESPACE_END
#endif
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class R, class S = std::error_code, class NoValuePolicy = policy::default_policy<R, S, void>> //
using result = std_result<R, S, NoValuePolicy>;
/*! AWAITING HUGO JSON CONVERSION TOOL
type alias template <class R, class S = std::error_code> unchecked. Potential doc page: `unchecked<T, E = varies>`
*/
template <class R, class S = std::error_code> using unchecked = result<R, S, policy::all_narrow>;
/*! AWAITING HUGO JSON CONVERSION TOOL
type alias template <class R, class S = std::error_code> checked. Potential doc page: `checked<T, E = varies>`
*/
template <class R, class S = std::error_code> using checked = result<R, S, policy::throw_bad_result_access<S, void>>;
OUTCOME_V2_NAMESPACE_END
#endif
/* A less simple result type
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (6 commits)
File Created: June 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_STD_OUTCOME_HPP
#define OUTCOME_STD_OUTCOME_HPP
/* A less simple result type
(C) 2017-2020 Niall Douglas <http://www.nedproductions.biz/> (20 commits)
File Created: June 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_BASIC_OUTCOME_HPP
#define OUTCOME_BASIC_OUTCOME_HPP
/* Exception observers for outcome type
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (3 commits)
File Created: Oct 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_BASIC_OUTCOME_EXCEPTION_OBSERVERS_HPP
#define OUTCOME_BASIC_OUTCOME_EXCEPTION_OBSERVERS_HPP
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace detail
{
  template <class Base, class R, class S, class P, class NoValuePolicy> class basic_outcome_exception_observers : public Base
  {
  public:
    using exception_type = P;
    using Base::Base;
    constexpr inline exception_type &assume_exception() & noexcept;
    constexpr inline const exception_type &assume_exception() const &noexcept;
    constexpr inline exception_type &&assume_exception() && noexcept;
    constexpr inline const exception_type &&assume_exception() const &&noexcept;
    constexpr inline exception_type &exception() &;
    constexpr inline const exception_type &exception() const &;
    constexpr inline exception_type &&exception() &&;
    constexpr inline const exception_type &&exception() const &&;
  };
  // Exception observers not present
  template <class Base, class R, class S, class NoValuePolicy> class basic_outcome_exception_observers<Base, R, S, void, NoValuePolicy> : public Base
  {
  public:
    using Base::Base;
    constexpr void assume_exception() const noexcept { NoValuePolicy::narrow_exception_check(this); }
    constexpr void exception() const { NoValuePolicy::wide_exception_check(this); }
  };
} // namespace detail
OUTCOME_V2_NAMESPACE_END
#endif
/* Failure observers for outcome type
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (7 commits)
File Created: Oct 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_BASIC_OUTCOME_FAILURE_OBSERVERS_HPP
#define OUTCOME_BASIC_OUTCOME_FAILURE_OBSERVERS_HPP
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace detail
{
  namespace adl
  {
    struct search_detail_adl
    {
    };
    // Do NOT use template requirements here!
    template <class S, typename = decltype(basic_outcome_failure_exception_from_error(std::declval<S>()))>
    inline auto _delayed_lookup_basic_outcome_failure_exception_from_error(const S &ec, search_detail_adl /*unused*/)
    {
      // ADL discovered
      return basic_outcome_failure_exception_from_error(ec);
    }
  } // namespace adl
#if defined(_MSC_VER) && _MSC_VER <= 1923 // VS2019
  // VS2017 and VS2019 with /permissive- chokes on the correct form due to over eager early instantiation.
  template <class S, class P> inline void _delayed_lookup_basic_outcome_failure_exception_from_error(...) { static_assert(sizeof(S) == 0, "No specialisation for these error and exception types available!"); }
#else
  template <class S, class P> inline void _delayed_lookup_basic_outcome_failure_exception_from_error(...) = delete; // NOLINT No specialisation for these error and exception types available!
#endif
  template <class exception_type> inline exception_type current_exception_or_fatal(std::exception_ptr e) { std::rethrow_exception(e); }
  template <> inline std::exception_ptr current_exception_or_fatal<std::exception_ptr>(std::exception_ptr e) { return e; }
  template <class Base, class R, class S, class P, class NoValuePolicy> class basic_outcome_failure_observers : public Base
  {
  public:
    using exception_type = P;
    using Base::Base;
    exception_type failure() const noexcept
    {
#ifdef __cpp_exceptions
      try
#endif
      {
        if(this->_state._status.have_exception())
        {
          return this->assume_exception();
        }
        if(this->_state._status.have_error())
        {
          return _delayed_lookup_basic_outcome_failure_exception_from_error(this->assume_error(), adl::search_detail_adl());
        }
        return exception_type();
      }
#ifdef __cpp_exceptions
      catch(...)
      {
        // Return the failure if exception_type is std::exception_ptr,
        // otherwise terminate same as throwing an exception inside noexcept
        return current_exception_or_fatal<exception_type>(std::current_exception());
      }
#endif
    }
  };
} // namespace detail
OUTCOME_V2_NAMESPACE_END
#endif
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation" // Standardese markup confuses clang
#endif
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
template <class R, class S, class P, class NoValuePolicy> //
class basic_outcome;
namespace detail
{
  // May be reused by basic_outcome subclasses to save load on the compiler
  template <class value_type, class error_type, class exception_type> struct outcome_predicates
  {
    using result = result_predicates<value_type, error_type>;
    // Predicate for the implicit constructors to be available
    static constexpr bool implicit_constructors_enabled = //
    result::implicit_constructors_enabled //
    && !detail::is_implicitly_constructible<value_type, exception_type> //
    && !detail::is_implicitly_constructible<error_type, exception_type> //
    && !detail::is_implicitly_constructible<exception_type, value_type> //
    && !detail::is_implicitly_constructible<exception_type, error_type>;
    // Predicate for the value converting constructor to be available.
    template <class T>
    static constexpr bool enable_value_converting_constructor = //
    implicit_constructors_enabled //
    &&result::template enable_value_converting_constructor<T> //
    && !detail::is_implicitly_constructible<exception_type, T>; // deliberately less tolerant of ambiguity than result's edition
    // Predicate for the error converting constructor to be available.
    template <class T>
    static constexpr bool enable_error_converting_constructor = //
    implicit_constructors_enabled //
    &&result::template enable_error_converting_constructor<T> //
    && !detail::is_implicitly_constructible<exception_type, T>; // deliberately less tolerant of ambiguity than result's edition
    // Predicate for the error condition converting constructor to be available.
    template <class ErrorCondEnum>
    static constexpr bool enable_error_condition_converting_constructor = result::template enable_error_condition_converting_constructor<ErrorCondEnum> //
                                                                          && !detail::is_implicitly_constructible<exception_type, ErrorCondEnum>;
    // Predicate for the exception converting constructor to be available.
    template <class T>
    static constexpr bool enable_exception_converting_constructor = //
    implicit_constructors_enabled //
    && !is_in_place_type_t<std::decay_t<T>>::value // not in place construction
    && !detail::is_implicitly_constructible<value_type, T> && !detail::is_implicitly_constructible<error_type, T> &&
    detail::is_implicitly_constructible<exception_type, T>;
    // Predicate for the error + exception converting constructor to be available.
    template <class T, class U>
    static constexpr bool enable_error_exception_converting_constructor = //
    implicit_constructors_enabled //
    && !is_in_place_type_t<std::decay_t<T>>::value // not in place construction
    && !detail::is_implicitly_constructible<value_type, T> && detail::is_implicitly_constructible<error_type, T> //
    && !detail::is_implicitly_constructible<value_type, U> && detail::is_implicitly_constructible<exception_type, U>;
    // Predicate for the converting copy constructor from a compatible outcome to be available.
    template <class T, class U, class V, class W>
    static constexpr bool enable_compatible_conversion = //
    (std::is_void<T>::value ||
     detail::is_explicitly_constructible<value_type, typename basic_outcome<T, U, V, W>::value_type>) // if our value types are constructible
    &&(std::is_void<U>::value ||
       detail::is_explicitly_constructible<error_type, typename basic_outcome<T, U, V, W>::error_type>) // if our error types are constructible
    &&(std::is_void<V>::value ||
       detail::is_explicitly_constructible<exception_type, typename basic_outcome<T, U, V, W>::exception_type>) // if our exception types are constructible
    ;
    // Predicate for the converting constructor from a make_error_code() of the input to be available.
    template <class T, class U, class V, class W>
    static constexpr bool enable_make_error_code_compatible_conversion = //
    trait::is_error_code_available<std::decay_t<error_type>>::value // if error type has an error code
    && !enable_compatible_conversion<T, U, V, W> // and the normal compatible conversion is not available
    && (std::is_void<T>::value ||
        detail::is_explicitly_constructible<value_type, typename basic_outcome<T, U, V, W>::value_type>) // and if our value types are constructible
    &&detail::is_explicitly_constructible<error_type,
                                          typename trait::is_error_code_available<U>::type> // and our error type is constructible from a make_error_code()
    && (std::is_void<V>::value ||
        detail::is_explicitly_constructible<exception_type, typename basic_outcome<T, U, V, W>::exception_type>); // and our exception types are constructible
    // Predicate for the implicit converting inplace constructor from a compatible input to be available.
    struct disable_inplace_value_error_exception_constructor;
    template <class... Args>
    using choose_inplace_value_error_exception_constructor = std::conditional_t< //
    ((static_cast<int>(detail::is_constructible<value_type, Args...>) + static_cast<int>(detail::is_constructible<error_type, Args...>) +
      static_cast<int>(detail::is_constructible<exception_type, Args...>)) > 1), //
    disable_inplace_value_error_exception_constructor, //
    std::conditional_t< //
    detail::is_constructible<value_type, Args...>, //
    value_type, //
    std::conditional_t< //
    detail::is_constructible<error_type, Args...>, //
    error_type, //
    std::conditional_t< //
    detail::is_constructible<exception_type, Args...>, //
    exception_type, //
    disable_inplace_value_error_exception_constructor>>>>;
    template <class... Args>
    static constexpr bool enable_inplace_value_error_exception_constructor = //
    implicit_constructors_enabled &&
    !std::is_same<choose_inplace_value_error_exception_constructor<Args...>, disable_inplace_value_error_exception_constructor>::value;
  };
  // Select whether to use basic_outcome_failure_observers or not
  template <class Base, class R, class S, class P, class NoValuePolicy>
  using select_basic_outcome_failure_observers = //
  std::conditional_t<trait::is_error_code_available<S>::value && trait::is_exception_ptr_available<P>::value,
                     basic_outcome_failure_observers<Base, R, S, P, NoValuePolicy>, Base>;
  template <class T, class U, class V> constexpr inline const V &extract_exception_from_failure(const failure_type<U, V> &v) { return v.exception(); }
  template <class T, class U, class V> constexpr inline V &&extract_exception_from_failure(failure_type<U, V> &&v)
  {
    return static_cast<failure_type<U, V> &&>(v).exception();
  }
  template <class T, class U> constexpr inline const U &extract_exception_from_failure(const failure_type<U, void> &v) { return v.error(); }
  template <class T, class U> constexpr inline U &&extract_exception_from_failure(failure_type<U, void> &&v)
  {
    return static_cast<failure_type<U, void> &&>(v).error();
  }
  template <class T> struct is_basic_outcome
  {
    static constexpr bool value = false;
  };
  template <class R, class S, class T, class N> struct is_basic_outcome<basic_outcome<R, S, T, N>>
  {
    static constexpr bool value = true;
  };
} // namespace detail
/*! AWAITING HUGO JSON CONVERSION TOOL
type alias template <class T> is_basic_outcome. Potential doc page: `is_basic_outcome<T>`
*/
template <class T> using is_basic_outcome = detail::is_basic_outcome<std::decay_t<T>>;
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class T> static constexpr bool is_basic_outcome_v = detail::is_basic_outcome<std::decay_t<T>>::value;
namespace concepts
{
#if defined(__cpp_concepts)
  /* The `basic_outcome` concept.
  \requires That `U` matches a `basic_outcome`.
  */
  template <class U>
  concept OUTCOME_GCC6_CONCEPT_BOOL basic_outcome =
  OUTCOME_V2_NAMESPACE::is_basic_outcome<U>::value ||
  (requires(U v) {
    OUTCOME_V2_NAMESPACE::basic_outcome<typename U::value_type, typename U::error_type, typename U::exception_type, typename U::no_value_policy_type>(v);
  } && //
   detail::convertible<
   U, OUTCOME_V2_NAMESPACE::basic_outcome<typename U::value_type, typename U::error_type, typename U::exception_type, typename U::no_value_policy_type>> && //
   detail::base_of<
   OUTCOME_V2_NAMESPACE::basic_outcome<typename U::value_type, typename U::error_type, typename U::exception_type, typename U::no_value_policy_type>, U>);
#else
  namespace detail
  {
    inline no_match match_basic_outcome(...);
    template <class R, class S, class P, class NVP, class T, //
              typename = typename T::value_type, //
              typename = typename T::error_type, //
              typename = typename T::exception_type, //
              typename = typename T::no_value_policy_type, //
              typename std::enable_if_t<std::is_convertible<T, OUTCOME_V2_NAMESPACE::basic_outcome<R, S, P, NVP>>::value && //
                                        std::is_base_of<OUTCOME_V2_NAMESPACE::basic_outcome<R, S, P, NVP>, T>::value,
                                        bool> = true>
    inline OUTCOME_V2_NAMESPACE::basic_outcome<R, S, P, NVP> match_basic_outcome(OUTCOME_V2_NAMESPACE::basic_outcome<R, S, P, NVP> &&, T &&);
    template <class U>
    static constexpr bool basic_outcome =
    OUTCOME_V2_NAMESPACE::is_basic_outcome<U>::value ||
    !std::is_same<no_match, decltype(match_basic_outcome(std::declval<OUTCOME_V2_NAMESPACE::detail::devoid<U>>(),
                                                         std::declval<OUTCOME_V2_NAMESPACE::detail::devoid<U>>()))>::value;
  } // namespace detail
  /* The `basic_outcome` concept.
  \requires That `U` matches a `basic_outcome`.
  */
  template <class U> static constexpr bool basic_outcome = detail::basic_outcome<U>;
#endif
} // namespace concepts
namespace hooks
{
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class R, class S, class P, class NoValuePolicy, class U>
  constexpr inline void override_outcome_exception(basic_outcome<R, S, P, NoValuePolicy> *o, U &&v) noexcept;
} // namespace hooks
/*! AWAITING HUGO JSON CONVERSION TOOL
type definition template <class R, class S, class P, class NoValuePolicy> basic_outcome. Potential doc page: `basic_outcome<T, EC, EP, NoValuePolicy>`
*/
template <class R, class S, class P, class NoValuePolicy> //
class OUTCOME_NODISCARD basic_outcome
    : public detail::select_basic_outcome_failure_observers<
      detail::basic_outcome_exception_observers<detail::basic_result_final<R, S, NoValuePolicy>, R, S, P, NoValuePolicy>, R, S, P, NoValuePolicy>
{
  static_assert(trait::type_can_be_used_in_basic_result<P>, "The exception_type cannot be used");
  static_assert(std::is_void<P>::value || std::is_default_constructible<P>::value, "exception_type must be void or default constructible");
  using base = detail::select_basic_outcome_failure_observers<
  detail::basic_outcome_exception_observers<detail::basic_result_final<R, S, NoValuePolicy>, R, S, P, NoValuePolicy>, R, S, P, NoValuePolicy>;
  friend struct policy::base;
  template <class T, class U, class V, class W> //
  friend class basic_outcome;
  template <class T, class U, class V, class W, class X>
  friend constexpr inline void hooks::override_outcome_exception(basic_outcome<T, U, V, W> *o, X &&v) noexcept; // NOLINT
  struct implicit_constructors_disabled_tag
  {
  };
  struct value_converting_constructor_tag
  {
  };
  struct error_converting_constructor_tag
  {
  };
  struct error_condition_converting_constructor_tag
  {
  };
  struct exception_converting_constructor_tag
  {
  };
  struct error_exception_converting_constructor_tag
  {
  };
  struct explicit_valueorerror_converting_constructor_tag
  {
  };
  struct explicit_compatible_copy_conversion_tag
  {
  };
  struct explicit_compatible_move_conversion_tag
  {
  };
  struct explicit_make_error_code_compatible_copy_conversion_tag
  {
  };
  struct explicit_make_error_code_compatible_move_conversion_tag
  {
  };
  struct error_failure_tag
  {
  };
  struct exception_failure_tag
  {
  };
  struct disable_in_place_value_type
  {
  };
  struct disable_in_place_error_type
  {
  };
  struct disable_in_place_exception_type
  {
  };
public:
  using value_type = R;
  using error_type = S;
  using exception_type = P;
  using no_value_policy_type = NoValuePolicy;
  template <class T, class U = S, class V = P, class W = NoValuePolicy> using rebind = basic_outcome<T, U, V, W>;
protected:
  // Requirement predicates for outcome.
  struct predicate
  {
    using base = detail::outcome_predicates<value_type, error_type, exception_type>;
    // Predicate for any constructors to be available at all
    static constexpr bool constructors_enabled =
    (!std::is_same<std::decay_t<value_type>, std::decay_t<error_type>>::value || (std::is_void<value_type>::value && std::is_void<error_type>::value)) //
    && (!std::is_same<std::decay_t<value_type>, std::decay_t<exception_type>>::value ||
        (std::is_void<value_type>::value && std::is_void<exception_type>::value)) //
    && (!std::is_same<std::decay_t<error_type>, std::decay_t<exception_type>>::value ||
        (std::is_void<error_type>::value && std::is_void<exception_type>::value)) //
    ;
    // Predicate for implicit constructors to be available at all
    static constexpr bool implicit_constructors_enabled = constructors_enabled && base::implicit_constructors_enabled;
    // Predicate for the value converting constructor to be available.
    template <class T>
    static constexpr bool enable_value_converting_constructor = //
    constructors_enabled //
    && !std::is_same<std::decay_t<T>, basic_outcome>::value // not my type
    && base::template enable_value_converting_constructor<T>;
    // Predicate for the error converting constructor to be available.
    template <class T>
    static constexpr bool enable_error_converting_constructor = //
    constructors_enabled //
    && !std::is_same<std::decay_t<T>, basic_outcome>::value // not my type
    && base::template enable_error_converting_constructor<T>;
    // Predicate for the error condition converting constructor to be available.
    template <class ErrorCondEnum>
    static constexpr bool enable_error_condition_converting_constructor = //
    constructors_enabled //
    && !std::is_same<std::decay_t<ErrorCondEnum>, basic_outcome>::value // not my type
    && base::template enable_error_condition_converting_constructor<ErrorCondEnum>;
    // Predicate for the exception converting constructor to be available.
    template <class T>
    static constexpr bool enable_exception_converting_constructor = //
    constructors_enabled //
    && !std::is_same<std::decay_t<T>, basic_outcome>::value // not my type
    && base::template enable_exception_converting_constructor<T>;
    // Predicate for the error + exception converting constructor to be available.
    template <class T, class U>
    static constexpr bool enable_error_exception_converting_constructor = //
    constructors_enabled //
    && !std::is_same<std::decay_t<T>, basic_outcome>::value // not my type
    && base::template enable_error_exception_converting_constructor<T, U>;
    // Predicate for the converting constructor from a compatible input to be available.
    template <class T, class U, class V, class W>
    static constexpr bool enable_compatible_conversion = //
    constructors_enabled //
    && !std::is_same<basic_outcome<T, U, V, W>, basic_outcome>::value // not my type
    && base::template enable_compatible_conversion<T, U, V, W>;
    // Predicate for the converting constructor from a make_error_code() of the input to be available.
    template <class T, class U, class V, class W>
    static constexpr bool enable_make_error_code_compatible_conversion = //
    constructors_enabled //
    && !std::is_same<basic_outcome<T, U, V, W>, basic_outcome>::value // not my type
    && base::template enable_make_error_code_compatible_conversion<T, U, V, W>;
    // Predicate for the inplace construction of value to be available.
    template <class... Args>
    static constexpr bool enable_inplace_value_constructor = //
    constructors_enabled //
    && (std::is_void<value_type>::value //
        || detail::is_constructible<value_type, Args...>);
    // Predicate for the inplace construction of error to be available.
    template <class... Args>
    static constexpr bool enable_inplace_error_constructor = //
    constructors_enabled //
    && (std::is_void<error_type>::value //
        || detail::is_constructible<error_type, Args...>);
    // Predicate for the inplace construction of exception to be available.
    template <class... Args>
    static constexpr bool enable_inplace_exception_constructor = //
    constructors_enabled //
    && (std::is_void<exception_type>::value //
        || detail::is_constructible<exception_type, Args...>);
    // Predicate for the implicit converting inplace constructor to be available.
    template <class... Args>
    static constexpr bool enable_inplace_value_error_exception_constructor = //
    constructors_enabled //
    &&base::template enable_inplace_value_error_exception_constructor<Args...>;
    template <class... Args>
    using choose_inplace_value_error_exception_constructor = typename base::template choose_inplace_value_error_exception_constructor<Args...>;
  };
public:
  using value_type_if_enabled =
  std::conditional_t<std::is_same<value_type, error_type>::value || std::is_same<value_type, exception_type>::value, disable_in_place_value_type, value_type>;
  using error_type_if_enabled =
  std::conditional_t<std::is_same<error_type, value_type>::value || std::is_same<error_type, exception_type>::value, disable_in_place_error_type, error_type>;
  using exception_type_if_enabled = std::conditional_t<std::is_same<exception_type, value_type>::value || std::is_same<exception_type, error_type>::value,
                                                       disable_in_place_exception_type, exception_type>;
protected:
  detail::devoid<exception_type> _ptr;
public:
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class Arg, class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED((!predicate::constructors_enabled && sizeof...(Args) >= 0)))
  basic_outcome(Arg && /*unused*/, Args &&... /*unused*/) = delete; // NOLINT basic_outcome<> with any of the same type is NOT SUPPORTED, see docs!
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED((predicate::constructors_enabled && !predicate::implicit_constructors_enabled //
                                   && (detail::is_implicitly_constructible<value_type, T> || detail::is_implicitly_constructible<error_type, T> ||
                                       detail::is_implicitly_constructible<exception_type, T>) )))
  basic_outcome(T && /*unused*/, implicit_constructors_disabled_tag /*unused*/ = implicit_constructors_disabled_tag()) =
  delete; // NOLINT Implicit constructors disabled, use explicit in_place_type<T>, success() or failure(). see docs!
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_value_converting_constructor<T>))
  constexpr basic_outcome(T &&t, value_converting_constructor_tag /*unused*/ = value_converting_constructor_tag()) noexcept(
  detail::is_nothrow_constructible<value_type, T>) // NOLINT
      : base{in_place_type<typename base::_value_type>, static_cast<T &&>(t)}
      , _ptr()
  {
    no_value_policy_type::on_outcome_construction(this, static_cast<T &&>(t));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_error_converting_constructor<T>))
  constexpr basic_outcome(T &&t, error_converting_constructor_tag /*unused*/ = error_converting_constructor_tag()) noexcept(
  detail::is_nothrow_constructible<error_type, T>) // NOLINT
      : base{in_place_type<typename base::_error_type>, static_cast<T &&>(t)}
      , _ptr()
  {
    no_value_policy_type::on_outcome_construction(this, static_cast<T &&>(t));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class ErrorCondEnum)
  OUTCOME_TREQUIRES(OUTCOME_TEXPR(error_type(make_error_code(ErrorCondEnum()))), //
                    OUTCOME_TPRED(predicate::template enable_error_condition_converting_constructor<ErrorCondEnum>))
  constexpr basic_outcome(ErrorCondEnum &&t, error_condition_converting_constructor_tag /*unused*/ = error_condition_converting_constructor_tag()) noexcept(
  noexcept(error_type(make_error_code(static_cast<ErrorCondEnum &&>(t))))) // NOLINT
      : base{in_place_type<typename base::_error_type>, make_error_code(t)}
  {
    no_value_policy_type::on_outcome_construction(this, static_cast<ErrorCondEnum &&>(t));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_exception_converting_constructor<T>))
  constexpr basic_outcome(T &&t, exception_converting_constructor_tag /*unused*/ = exception_converting_constructor_tag()) noexcept(
  detail::is_nothrow_constructible<exception_type, T>) // NOLINT
      : base()
      , _ptr(static_cast<T &&>(t))
  {
    this->_state._status.set_have_exception(true);
    no_value_policy_type::on_outcome_construction(this, static_cast<T &&>(t));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_error_exception_converting_constructor<T, U>))
  constexpr basic_outcome(T &&a, U &&b, error_exception_converting_constructor_tag /*unused*/ = error_exception_converting_constructor_tag()) noexcept(
  detail::is_nothrow_constructible<error_type, T> &&detail::is_nothrow_constructible<exception_type, U>) // NOLINT
      : base{in_place_type<typename base::_error_type>, static_cast<T &&>(a)}
      , _ptr(static_cast<U &&>(b))
  {
    this->_state._status.set_have_exception(true);
    no_value_policy_type::on_outcome_construction(this, static_cast<T &&>(a), static_cast<U &&>(b));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(convert::value_or_error<basic_outcome, std::decay_t<T>>::enable_result_inputs || !concepts::basic_result<T>), //
                    OUTCOME_TPRED(convert::value_or_error<basic_outcome, std::decay_t<T>>::enable_outcome_inputs || !concepts::basic_outcome<T>), //
                    OUTCOME_TEXPR(convert::value_or_error<basic_outcome, std::decay_t<T>>{}(std::declval<T>())))
  constexpr explicit basic_outcome(T &&o,
                                   explicit_valueorerror_converting_constructor_tag /*unused*/ = explicit_valueorerror_converting_constructor_tag()) // NOLINT
      : basic_outcome{convert::value_or_error<basic_outcome, std::decay_t<T>>{}(static_cast<T &&>(o))}
  {
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V, class W)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_compatible_conversion<T, U, V, W>))
  constexpr explicit basic_outcome(
  const basic_outcome<T, U, V, W> &o,
  explicit_compatible_copy_conversion_tag /*unused*/ =
  explicit_compatible_copy_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T> &&detail::is_nothrow_constructible<error_type, U>
                                                      &&detail::is_nothrow_constructible<exception_type, V>)
      : base{typename base::compatible_conversion_tag(), o}
      , _ptr(o._ptr)
  {
    no_value_policy_type::on_outcome_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V, class W)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_compatible_conversion<T, U, V, W>))
  constexpr explicit basic_outcome(
  basic_outcome<T, U, V, W> &&o,
  explicit_compatible_move_conversion_tag /*unused*/ =
  explicit_compatible_move_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T> &&detail::is_nothrow_constructible<error_type, U>
                                                      &&detail::is_nothrow_constructible<exception_type, V>)
      : base{typename base::compatible_conversion_tag(), static_cast<basic_outcome<T, U, V, W> &&>(o)}
      , _ptr(static_cast<typename basic_outcome<T, U, V, W>::exception_type &&>(o._ptr))
  {
    no_value_policy_type::on_outcome_move_construction(this, static_cast<basic_outcome<T, U, V, W> &&>(o));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(detail::result_predicates<value_type, error_type>::template enable_compatible_conversion<T, U, V>))
  constexpr explicit basic_outcome(
  const basic_result<T, U, V> &o,
  explicit_compatible_copy_conversion_tag /*unused*/ =
  explicit_compatible_copy_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T> &&detail::is_nothrow_constructible<error_type, U>
                                                      &&detail::is_nothrow_constructible<exception_type>)
      : base{typename base::compatible_conversion_tag(), o}
      , _ptr()
  {
    no_value_policy_type::on_outcome_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(detail::result_predicates<value_type, error_type>::template enable_compatible_conversion<T, U, V>))
  constexpr explicit basic_outcome(
  basic_result<T, U, V> &&o,
  explicit_compatible_move_conversion_tag /*unused*/ =
  explicit_compatible_move_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T> &&detail::is_nothrow_constructible<error_type, U>
                                                      &&detail::is_nothrow_constructible<exception_type>)
      : base{typename base::compatible_conversion_tag(), static_cast<basic_result<T, U, V> &&>(o)}
      , _ptr()
  {
    no_value_policy_type::on_outcome_move_construction(this, static_cast<basic_result<T, U, V> &&>(o));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(detail::result_predicates<value_type, error_type>::template enable_make_error_code_compatible_conversion<T, U, V>))
  constexpr explicit basic_outcome(const basic_result<T, U, V> &o,
                                   explicit_make_error_code_compatible_copy_conversion_tag /*unused*/ =
                                   explicit_make_error_code_compatible_copy_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T>
                                                                                                       &&noexcept(make_error_code(std::declval<U>())) &&
                                                                                                       detail::is_nothrow_constructible<exception_type>)
      : base{typename base::make_error_code_compatible_conversion_tag(), o}
      , _ptr()
  {
    no_value_policy_type::on_outcome_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(detail::result_predicates<value_type, error_type>::template enable_make_error_code_compatible_conversion<T, U, V>))
  constexpr explicit basic_outcome(basic_result<T, U, V> &&o,
                                   explicit_make_error_code_compatible_move_conversion_tag /*unused*/ =
                                   explicit_make_error_code_compatible_move_conversion_tag()) noexcept(detail::is_nothrow_constructible<value_type, T>
                                                                                                       &&noexcept(make_error_code(std::declval<U>())) &&
                                                                                                       detail::is_nothrow_constructible<exception_type>)
      : base{typename base::make_error_code_compatible_conversion_tag(), static_cast<basic_result<T, U, V> &&>(o)}
      , _ptr()
  {
    no_value_policy_type::on_outcome_move_construction(this, static_cast<basic_result<T, U, V> &&>(o));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_inplace_value_constructor<Args...>))
  constexpr explicit basic_outcome(in_place_type_t<value_type_if_enabled> _, Args &&... args) noexcept(detail::is_nothrow_constructible<value_type, Args...>)
      : base{_, static_cast<Args &&>(args)...}
      , _ptr()
  {
    no_value_policy_type::on_outcome_in_place_construction(this, in_place_type<value_type>, static_cast<Args &&>(args)...);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class U, class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_inplace_value_constructor<std::initializer_list<U>, Args...>))
  constexpr explicit basic_outcome(in_place_type_t<value_type_if_enabled> _, std::initializer_list<U> il,
                                   Args &&... args) noexcept(detail::is_nothrow_constructible<value_type, std::initializer_list<U>, Args...>)
      : base{_, il, static_cast<Args &&>(args)...}
      , _ptr()
  {
    no_value_policy_type::on_outcome_in_place_construction(this, in_place_type<value_type>, il, static_cast<Args &&>(args)...);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_inplace_error_constructor<Args...>))
  constexpr explicit basic_outcome(in_place_type_t<error_type_if_enabled> _, Args &&... args) noexcept(detail::is_nothrow_constructible<error_type, Args...>)
      : base{_, static_cast<Args &&>(args)...}
      , _ptr()
  {
    no_value_policy_type::on_outcome_in_place_construction(this, in_place_type<error_type>, static_cast<Args &&>(args)...);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class U, class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_inplace_error_constructor<std::initializer_list<U>, Args...>))
  constexpr explicit basic_outcome(in_place_type_t<error_type_if_enabled> _, std::initializer_list<U> il,
                                   Args &&... args) noexcept(detail::is_nothrow_constructible<error_type, std::initializer_list<U>, Args...>)
      : base{_, il, static_cast<Args &&>(args)...}
      , _ptr()
  {
    no_value_policy_type::on_outcome_in_place_construction(this, in_place_type<error_type>, il, static_cast<Args &&>(args)...);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_inplace_exception_constructor<Args...>))
  constexpr explicit basic_outcome(in_place_type_t<exception_type_if_enabled> /*unused*/,
                                   Args &&... args) noexcept(detail::is_nothrow_constructible<exception_type, Args...>)
      : base()
      , _ptr(static_cast<Args &&>(args)...)
  {
    this->_state._status.set_have_exception(true);
    no_value_policy_type::on_outcome_in_place_construction(this, in_place_type<exception_type>, static_cast<Args &&>(args)...);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class U, class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_inplace_exception_constructor<std::initializer_list<U>, Args...>))
  constexpr explicit basic_outcome(in_place_type_t<exception_type_if_enabled> /*unused*/, std::initializer_list<U> il,
                                   Args &&... args) noexcept(detail::is_nothrow_constructible<exception_type, std::initializer_list<U>, Args...>)
      : base()
      , _ptr(il, static_cast<Args &&>(args)...)
  {
    this->_state._status.set_have_exception(true);
    no_value_policy_type::on_outcome_in_place_construction(this, in_place_type<exception_type>, il, static_cast<Args &&>(args)...);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class A1, class A2, class... Args)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(predicate::template enable_inplace_value_error_exception_constructor<A1, A2, Args...>))
  constexpr basic_outcome(A1 &&a1, A2 &&a2, Args &&... args) noexcept(
  noexcept(typename predicate::template choose_inplace_value_error_exception_constructor<A1, A2, Args...>(std::declval<A1>(), std::declval<A2>(),
                                                                                                          std::declval<Args>()...)))
      : basic_outcome(in_place_type<typename predicate::template choose_inplace_value_error_exception_constructor<A1, A2, Args...>>, static_cast<A1 &&>(a1),
                      static_cast<A2 &&>(a2), static_cast<Args &&>(args)...)
  {
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  constexpr basic_outcome(const success_type<void> &o) noexcept(std::is_nothrow_default_constructible<value_type>::value) // NOLINT
      : base{in_place_type<typename base::_value_type>}
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_outcome_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_compatible_conversion<T, void, void, void>))
  constexpr basic_outcome(const success_type<T> &o) noexcept(detail::is_nothrow_constructible<value_type, T>) // NOLINT
      : base{in_place_type<typename base::_value_type>, detail::extract_value_from_success<value_type>(o)}
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_outcome_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_compatible_conversion<T, void, void, void>))
  constexpr basic_outcome(success_type<T> &&o) noexcept(detail::is_nothrow_constructible<value_type, T>) // NOLINT
      : base{in_place_type<typename base::_value_type>, detail::extract_value_from_success<value_type>(static_cast<success_type<T> &&>(o))}
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_outcome_move_construction(this, static_cast<success_type<T> &&>(o));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_compatible_conversion<void, T, void, void>))
  constexpr basic_outcome(const failure_type<T> &o,
                          error_failure_tag /*unused*/ = error_failure_tag()) noexcept(detail::is_nothrow_constructible<error_type, T>) // NOLINT
      : base{in_place_type<typename base::_error_type>, detail::extract_error_from_failure<error_type>(o)}
      , _ptr()
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_outcome_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_compatible_conversion<void, void, T, void>))
  constexpr basic_outcome(const failure_type<T> &o,
                          exception_failure_tag /*unused*/ = exception_failure_tag()) noexcept(detail::is_nothrow_constructible<exception_type, T>) // NOLINT
      : base()
      , _ptr(detail::extract_exception_from_failure<exception_type>(o))
  {
    this->_state._status.set_have_exception(true);
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_outcome_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_make_error_code_compatible_conversion<void, T, void, void>))
  constexpr basic_outcome(const failure_type<T> &o,
                          explicit_make_error_code_compatible_copy_conversion_tag /*unused*/ =
                          explicit_make_error_code_compatible_copy_conversion_tag()) noexcept(noexcept(make_error_code(std::declval<T>()))) // NOLINT
      : base{in_place_type<typename base::_error_type>, make_error_code(detail::extract_error_from_failure<error_type>(o))}
      , _ptr()
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_outcome_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_void<U>::value && predicate::template enable_compatible_conversion<void, T, U, void>))
  constexpr basic_outcome(const failure_type<T, U> &o, explicit_compatible_copy_conversion_tag /*unused*/ = explicit_compatible_copy_conversion_tag()) noexcept(
  detail::is_nothrow_constructible<error_type, T> &&detail::is_nothrow_constructible<exception_type, U>) // NOLINT
      : base{in_place_type<typename base::_error_type>, detail::extract_error_from_failure<error_type>(o)}
      , _ptr(detail::extract_exception_from_failure<exception_type>(o))
  {
    if(!o.has_error())
    {
      this->_state._status.set_have_error(false);
    }
    if(o.has_exception())
    {
      this->_state._status.set_have_exception(true);
    }
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_outcome_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_compatible_conversion<void, T, void, void>))
  constexpr basic_outcome(failure_type<T> &&o,
                          error_failure_tag /*unused*/ = error_failure_tag()) noexcept(detail::is_nothrow_constructible<error_type, T>) // NOLINT
      : base{in_place_type<typename base::_error_type>, detail::extract_error_from_failure<error_type>(static_cast<failure_type<T> &&>(o))}
      , _ptr()
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_outcome_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_compatible_conversion<void, void, T, void>))
  constexpr basic_outcome(failure_type<T> &&o,
                          exception_failure_tag /*unused*/ = exception_failure_tag()) noexcept(detail::is_nothrow_constructible<exception_type, T>) // NOLINT
      : base()
      , _ptr(detail::extract_exception_from_failure<exception_type>(static_cast<failure_type<T> &&>(o)))
  {
    this->_state._status.set_have_exception(true);
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_outcome_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_void<T>::value && predicate::template enable_make_error_code_compatible_conversion<void, T, void, void>))
  constexpr basic_outcome(failure_type<T> &&o,
                          explicit_make_error_code_compatible_move_conversion_tag /*unused*/ =
                          explicit_make_error_code_compatible_move_conversion_tag()) noexcept(noexcept(make_error_code(std::declval<T>()))) // NOLINT
      : base{in_place_type<typename base::_error_type>, make_error_code(detail::extract_error_from_failure<error_type>(static_cast<failure_type<T> &&>(o)))}
      , _ptr()
  {
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_outcome_copy_construction(this, o);
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_void<U>::value && predicate::template enable_compatible_conversion<void, T, U, void>))
  constexpr basic_outcome(failure_type<T, U> &&o, explicit_compatible_move_conversion_tag /*unused*/ = explicit_compatible_move_conversion_tag()) noexcept(
  detail::is_nothrow_constructible<error_type, T> &&detail::is_nothrow_constructible<exception_type, U>) // NOLINT
      : base{in_place_type<typename base::_error_type>, detail::extract_error_from_failure<error_type>(static_cast<failure_type<T, U> &&>(o))}
      , _ptr(detail::extract_exception_from_failure<exception_type>(static_cast<failure_type<T, U> &&>(o)))
  {
    if(!o.has_error())
    {
      this->_state._status.set_have_error(false);
    }
    if(o.has_exception())
    {
      this->_state._status.set_have_exception(true);
    }
    hooks::set_spare_storage(this, o.spare_storage());
    no_value_policy_type::on_outcome_move_construction(this, static_cast<failure_type<T, U> &&>(o));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  using base::operator==;
  using base::operator!=;
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V, class W)
  OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<detail::devoid<value_type>>() == std::declval<detail::devoid<T>>()), //
                    OUTCOME_TEXPR(std::declval<detail::devoid<error_type>>() == std::declval<detail::devoid<U>>()), //
                    OUTCOME_TEXPR(std::declval<detail::devoid<exception_type>>() == std::declval<detail::devoid<V>>()))
  constexpr bool operator==(const basic_outcome<T, U, V, W> &o) const noexcept( //
  noexcept(std::declval<detail::devoid<value_type>>() == std::declval<detail::devoid<T>>()) //
  &&noexcept(std::declval<detail::devoid<error_type>>() == std::declval<detail::devoid<U>>()) //
  &&noexcept(std::declval<detail::devoid<exception_type>>() == std::declval<detail::devoid<V>>()))
  {
    if(this->_state._status.have_value() && o._state._status.have_value())
    {
      return this->_state._value == o._state._value; // NOLINT
    }
    if(this->_state._status.have_error() && o._state._status.have_error() //
       && this->_state._status.have_exception() && o._state._status.have_exception())
    {
      return this->_state._error == o._state._error && this->_ptr == o._ptr;
    }
    if(this->_state._status.have_error() && o._state._status.have_error())
    {
      return this->_state._error == o._state._error;
    }
    if(this->_state._status.have_exception() && o._state._status.have_exception())
    {
      return this->_ptr == o._ptr;
    }
    return false;
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U)
  OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<error_type>() == std::declval<T>()), //
                    OUTCOME_TEXPR(std::declval<exception_type>() == std::declval<U>()))
  constexpr bool operator==(const failure_type<T, U> &o) const noexcept( //
  noexcept(std::declval<error_type>() == std::declval<T>()) &&noexcept(std::declval<exception_type>() == std::declval<U>()))
  {
    if(this->_state._status.have_error() && o._state._status.have_error() //
       && this->_state._status.have_exception() && o._state._status.have_exception())
    {
      return this->_state._error == o.error() && this->_ptr == o.exception();
    }
    if(this->_state._status.have_error() && o._state._status.have_error())
    {
      return this->_state._error == o.error();
    }
    if(this->_state._status.have_exception() && o._state._status.have_exception())
    {
      return this->_ptr == o.exception();
    }
    return false;
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U, class V, class W)
  OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<detail::devoid<value_type>>() != std::declval<detail::devoid<T>>()), //
                    OUTCOME_TEXPR(std::declval<detail::devoid<error_type>>() != std::declval<detail::devoid<U>>()), //
                    OUTCOME_TEXPR(std::declval<detail::devoid<exception_type>>() != std::declval<detail::devoid<V>>()))
  constexpr bool operator!=(const basic_outcome<T, U, V, W> &o) const noexcept( //
  noexcept(std::declval<detail::devoid<value_type>>() != std::declval<detail::devoid<T>>()) //
  &&noexcept(std::declval<detail::devoid<error_type>>() != std::declval<detail::devoid<U>>()) //
  &&noexcept(std::declval<detail::devoid<exception_type>>() != std::declval<detail::devoid<V>>()))
  {
    if(this->_state._status.have_value() && o._state._status.have_value())
    {
      return this->_state._value != o._state._value; // NOLINT
    }
    if(this->_state._status.have_error() && o._state._status.have_error() //
       && this->_state._status.have_exception() && o._state._status.have_exception())
    {
      return this->_state._error != o._state._error || this->_ptr != o._ptr;
    }
    if(this->_state._status.have_error() && o._state._status.have_error())
    {
      return this->_state._error != o._state._error;
    }
    if(this->_state._status.have_exception() && o._state._status.have_exception())
    {
      return this->_ptr != o._ptr;
    }
    return true;
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  OUTCOME_TEMPLATE(class T, class U)
  OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<error_type>() != std::declval<T>()), //
                    OUTCOME_TEXPR(std::declval<exception_type>() != std::declval<U>()))
  constexpr bool operator!=(const failure_type<T, U> &o) const noexcept( //
  noexcept(std::declval<error_type>() == std::declval<T>()) &&noexcept(std::declval<exception_type>() == std::declval<U>()))
  {
    if(this->_state._status.have_error() && o._state._status.have_error() //
       && this->_state._status.have_exception() && o._state._status.have_exception())
    {
      return this->_state._error != o.error() || this->_ptr != o.exception();
    }
    if(this->_state._status.have_error() && o._state._status.have_error())
    {
      return this->_state._error != o.error();
    }
    if(this->_state._status.have_exception() && o._state._status.have_exception())
    {
      return this->_ptr != o.exception();
    }
    return true;
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  constexpr void swap(basic_outcome &o) noexcept((std::is_void<value_type>::value || detail::is_nothrow_swappable<value_type>::value) //
                                                 && (std::is_void<error_type>::value || detail::is_nothrow_swappable<error_type>::value) //
                                                 && (std::is_void<exception_type>::value || detail::is_nothrow_swappable<exception_type>::value))
  {
#ifdef __cpp_exceptions
    constexpr bool value_throws = !std::is_void<value_type>::value && !detail::is_nothrow_swappable<value_type>::value;
    constexpr bool error_throws = !std::is_void<error_type>::value && !detail::is_nothrow_swappable<error_type>::value;
    constexpr bool exception_throws = !std::is_void<exception_type>::value && !detail::is_nothrow_swappable<exception_type>::value;
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4127) // conditional expression is constant
#endif
    if(!exception_throws && !value_throws && !error_throws)
    {
      // Simples
      this->_state.swap(o._state);
      using std::swap;
      swap(this->_ptr, o._ptr);
      return;
    }
    struct _
    {
      basic_outcome &a, &b;
      bool exceptioned{false};
      bool all_good{false};
      ~_()
      {
        if(!all_good)
        {
          // We lost one of the values
          a._state._status.set_have_lost_consistency(true);
          b._state._status.set_have_lost_consistency(true);
          return;
        }
        if(exceptioned)
        {
          // The value + error swap threw an exception. Try to swap back _ptr
          try
          {
            strong_swap(all_good, a._ptr, b._ptr);
          }
          catch(...)
          {
            // We lost one of the values
            a._state._status.set_have_lost_consistency(true);
            b._state._status.set_have_lost_consistency(true);
            // throw away second exception
          }
          // Prevent has_value() == has_error() or has_value() == has_exception()
          auto check = [](basic_outcome *t) {
            if(t->has_value() && (t->has_error() || t->has_exception()))
            {
              t->_state._status.set_have_error(false).set_have_exception(false);
              t->_state._status.set_have_lost_consistency(true);
            }
            if(!t->has_value() && !(t->has_error() || t->has_exception()))
            {
              // Choose error, for no particular reason
              t->_state._status.set_have_error(true).set_have_lost_consistency(true);
            }
          };
          check(&a);
          check(&b);
        }
      }
    } _{*this, o};
    strong_swap(_.all_good, this->_ptr, o._ptr);
    _.exceptioned = true;
    this->_state.swap(o._state);
    _.exceptioned = false;
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#else
    this->_state.swap(o._state);
    using std::swap;
    swap(this->_ptr, o._ptr);
#endif
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  failure_type<error_type, exception_type> as_failure() const &
  {
    if(this->has_error() && this->has_exception())
    {
      return failure_type<error_type, exception_type>(this->assume_error(), this->assume_exception(), hooks::spare_storage(this));
    }
    if(this->has_exception())
    {
      return failure_type<error_type, exception_type>(in_place_type<exception_type>, this->assume_exception(), hooks::spare_storage(this));
    }
    return failure_type<error_type, exception_type>(in_place_type<error_type>, this->assume_error(), hooks::spare_storage(this));
  }
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  failure_type<error_type, exception_type> as_failure() &&
  {
    this->_state._status.set_have_moved_from(true);
    if(this->has_error() && this->has_exception())
    {
      return failure_type<error_type, exception_type>(static_cast<S &&>(this->assume_error()), static_cast<P &&>(this->assume_exception()),
                                                      hooks::spare_storage(this));
    }
    if(this->has_exception())
    {
      return failure_type<error_type, exception_type>(in_place_type<exception_type>, static_cast<P &&>(this->assume_exception()), hooks::spare_storage(this));
    }
    return failure_type<error_type, exception_type>(in_place_type<error_type>, static_cast<S &&>(this->assume_error()), hooks::spare_storage(this));
  }
#ifdef __APPLE__
  failure_type<error_type, exception_type> _xcode_workaround_as_failure() &&;
#endif
};
// C++ 20 operator== rewriting should take care of this for us, indeed
// if we don't disable it, we cause Concept recursion to infinity!
#if __cplusplus < 202000 && !_HAS_CXX20
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
OUTCOME_TEMPLATE(class T, class U, class V, //
                 class R, class S, class P, class N)
OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<basic_outcome<R, S, P, N>>() == std::declval<basic_result<T, U, V>>()))
constexpr inline bool operator==(const basic_result<T, U, V> &a, const basic_outcome<R, S, P, N> &b) noexcept( //
noexcept(std::declval<basic_outcome<R, S, P, N>>() == std::declval<basic_result<T, U, V>>()))
{
  return b == a;
}
#endif
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
OUTCOME_TEMPLATE(class T, class U, class V, //
                 class R, class S, class P, class N)
OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<basic_outcome<R, S, P, N>>() != std::declval<basic_result<T, U, V>>()))
constexpr inline bool operator!=(const basic_result<T, U, V> &a, const basic_outcome<R, S, P, N> &b) noexcept( //
noexcept(std::declval<basic_outcome<R, S, P, N>>() != std::declval<basic_result<T, U, V>>()))
{
  return b != a;
}
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class R, class S, class P, class N> inline void swap(basic_outcome<R, S, P, N> &a, basic_outcome<R, S, P, N> &b) noexcept(noexcept(a.swap(b)))
{
  a.swap(b);
}
namespace hooks
{
  /*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
  template <class R, class S, class P, class NoValuePolicy, class U>
  constexpr inline void override_outcome_exception(basic_outcome<R, S, P, NoValuePolicy> *o, U &&v) noexcept
  {
    o->_ptr = static_cast<U &&>(v); // NOLINT
    o->_state._status.set_have_exception(true);
  }
} // namespace hooks
OUTCOME_V2_NAMESPACE_END
#ifdef __clang__
#pragma clang diagnostic pop
#endif
/* Exception observers for outcome type
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (6 commits)
File Created: Oct 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_BASIC_OUTCOME_EXCEPTION_OBSERVERS_IMPL_HPP
#define OUTCOME_BASIC_OUTCOME_EXCEPTION_OBSERVERS_IMPL_HPP
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace policy
{
  template <class R, class S, class P, class NoValuePolicy, class Impl> inline constexpr auto &&base::_exception(Impl &&self) noexcept
  {
    // Impl will be some internal implementation class which has no knowledge of the _ptr stored
    // beneath it. So statically cast, preserving rvalue and constness, to the derived class.
    using Outcome = OUTCOME_V2_NAMESPACE::detail::rebind_type<basic_outcome<R, S, P, NoValuePolicy>, decltype(self)>;
#if defined(_MSC_VER) && _MSC_VER < 1920
    // VS2017 tries a copy construction in the correct implementation despite that Outcome is always a rvalue or lvalue ref! :(
    basic_outcome<R, S, P, NoValuePolicy> &_self = (basic_outcome<R, S, P, NoValuePolicy> &) (self); // NOLINT
#else
    Outcome _self = static_cast<Outcome>(self); // NOLINT
#endif
    return static_cast<Outcome>(_self)._ptr;
  }
} // namespace policy
namespace detail
{
  template <class Base, class R, class S, class P, class NoValuePolicy> inline constexpr typename basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::exception_type &basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::assume_exception() & noexcept
  {
    NoValuePolicy::narrow_exception_check(*this);
    return NoValuePolicy::template _exception<R, S, P, NoValuePolicy>(*this);
  }
  template <class Base, class R, class S, class P, class NoValuePolicy> inline constexpr const typename basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::exception_type &basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::assume_exception() const &noexcept
  {
    NoValuePolicy::narrow_exception_check(*this);
    return NoValuePolicy::template _exception<R, S, P, NoValuePolicy>(*this);
  }
  template <class Base, class R, class S, class P, class NoValuePolicy> inline constexpr typename basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::exception_type &&basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::assume_exception() && noexcept
  {
    NoValuePolicy::narrow_exception_check(std::move(*this));
    return NoValuePolicy::template _exception<R, S, P, NoValuePolicy>(std::move(*this));
  }
  template <class Base, class R, class S, class P, class NoValuePolicy> inline constexpr const typename basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::exception_type &&basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::assume_exception() const &&noexcept
  {
    NoValuePolicy::narrow_exception_check(std::move(*this));
    return NoValuePolicy::template _exception<R, S, P, NoValuePolicy>(std::move(*this));
  }
  template <class Base, class R, class S, class P, class NoValuePolicy> inline constexpr typename basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::exception_type &basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::exception() &
  {
    NoValuePolicy::wide_exception_check(*this);
    return NoValuePolicy::template _exception<R, S, P, NoValuePolicy>(*this);
  }
  template <class Base, class R, class S, class P, class NoValuePolicy> inline constexpr const typename basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::exception_type &basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::exception() const &
  {
    NoValuePolicy::wide_exception_check(*this);
    return NoValuePolicy::template _exception<R, S, P, NoValuePolicy>(*this);
  }
  template <class Base, class R, class S, class P, class NoValuePolicy> inline constexpr typename basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::exception_type &&basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::exception() &&
  {
    NoValuePolicy::wide_exception_check(std::move(*this));
    return NoValuePolicy::template _exception<R, S, P, NoValuePolicy>(std::move(*this));
  }
  template <class Base, class R, class S, class P, class NoValuePolicy> inline constexpr const typename basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::exception_type &&basic_outcome_exception_observers<Base, R, S, P, NoValuePolicy>::exception() const &&
  {
    NoValuePolicy::wide_exception_check(std::move(*this));
    return NoValuePolicy::template _exception<R, S, P, NoValuePolicy>(std::move(*this));
  }
} // namespace detail
OUTCOME_V2_NAMESPACE_END
#endif
#if !defined(NDEBUG)
OUTCOME_V2_NAMESPACE_BEGIN
// Check is trivial in all ways except default constructibility and standard layout
// static_assert(std::is_trivial<basic_outcome<int, long, double, policy::all_narrow>>::value, "outcome<int> is not trivial!");
// static_assert(std::is_trivially_default_constructible<basic_outcome<int, long, double, policy::all_narrow>>::value, "outcome<int> is not trivially default
// constructible!");
static_assert(std::is_trivially_copyable<basic_outcome<int, long, double, policy::all_narrow>>::value, "outcome<int> is not trivially copyable!");
static_assert(std::is_trivially_assignable<basic_outcome<int, long, double, policy::all_narrow>, basic_outcome<int, long, double, policy::all_narrow>>::value,
              "outcome<int> is not trivially assignable!");
static_assert(std::is_trivially_destructible<basic_outcome<int, long, double, policy::all_narrow>>::value, "outcome<int> is not trivially destructible!");
static_assert(std::is_trivially_copy_constructible<basic_outcome<int, long, double, policy::all_narrow>>::value,
              "outcome<int> is not trivially copy constructible!");
static_assert(std::is_trivially_move_constructible<basic_outcome<int, long, double, policy::all_narrow>>::value,
              "outcome<int> is not trivially move constructible!");
static_assert(std::is_trivially_copy_assignable<basic_outcome<int, long, double, policy::all_narrow>>::value, "outcome<int> is not trivially copy assignable!");
static_assert(std::is_trivially_move_assignable<basic_outcome<int, long, double, policy::all_narrow>>::value, "outcome<int> is not trivially move assignable!");
// Can't be standard layout as non-static member data is defined in more than one inherited class
// static_assert(std::is_standard_layout<basic_outcome<int, long, double, policy::all_narrow>>::value, "outcome<int> is not a standard layout type!");
OUTCOME_V2_NAMESPACE_END
#endif
#endif
#ifndef STD_BASIC_OUTCOME_FAILURE_EXCEPTION_FROM_ERROR
#define STD_BASIC_OUTCOME_FAILURE_EXCEPTION_FROM_ERROR
namespace std // NOLINT
{
  inline exception_ptr basic_outcome_failure_exception_from_error(const error_code &ec) { return make_exception_ptr(system_error(ec)); }
} // namespace std
#endif
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class R, class S = std::error_code, class P = std::exception_ptr, class NoValuePolicy = policy::default_policy<R, S, P>> //
using std_outcome = basic_outcome<R, S, P, NoValuePolicy>;
OUTCOME_V2_NAMESPACE_END
/* Policies for result and outcome
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (12 commits)
File Created: Oct 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_POLICY_OUTCOME_ERROR_CODE_THROW_AS_SYSTEM_ERROR_HPP
#define OUTCOME_POLICY_OUTCOME_ERROR_CODE_THROW_AS_SYSTEM_ERROR_HPP
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace policy
{
  /*! AWAITING HUGO JSON CONVERSION TOOL
type definition  error_code_throw_as_system_error. Potential doc page: NOT FOUND
*/
  template <class T, class EC, class E> struct error_code_throw_as_system_error : base
  {
    template <class Impl> static constexpr void wide_value_check(Impl &&self)
    {
      if(!base::_has_value(std::forward<Impl>(self)))
      {
        if(base::_has_exception(std::forward<Impl>(self)))
        {
          detail::_rethrow_exception<trait::is_exception_ptr_available<E>::value>{base::_exception<T, EC, E, error_code_throw_as_system_error>(std::forward<Impl>(self))}; // NOLINT
        }
        if(base::_has_error(std::forward<Impl>(self)))
        {
          // ADL discovered
          outcome_throw_as_system_error_with_payload(base::_error(std::forward<Impl>(self)));
        }
        OUTCOME_THROW_EXCEPTION(bad_outcome_access("no value")); // NOLINT
      }
    }
    template <class Impl> static constexpr void wide_error_check(Impl &&self)
    {
      if(!base::_has_error(std::forward<Impl>(self)))
      {
        OUTCOME_THROW_EXCEPTION(bad_outcome_access("no error")); // NOLINT
      }
    }
    template <class Impl> static constexpr void wide_exception_check(Impl &&self)
    {
      if(!base::_has_exception(std::forward<Impl>(self)))
      {
        OUTCOME_THROW_EXCEPTION(bad_outcome_access("no exception")); // NOLINT
      }
    }
  };
} // namespace policy
OUTCOME_V2_NAMESPACE_END
#endif
/* Policies for result and outcome
(C) 2017-2019 Niall Douglas <http://www.nedproductions.biz/> (10 commits)
File Created: Oct 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_POLICY_OUTCOME_EXCEPTION_PTR_RETHROW_HPP
#define OUTCOME_POLICY_OUTCOME_EXCEPTION_PTR_RETHROW_HPP
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
namespace policy
{
  /*! AWAITING HUGO JSON CONVERSION TOOL
type definition  exception_ptr_rethrow. Potential doc page: NOT FOUND
*/
  template <class T, class EC, class E> struct exception_ptr_rethrow : base
  {
    template <class Impl> static constexpr void wide_value_check(Impl &&self)
    {
      if(!base::_has_value(std::forward<Impl>(self)))
      {
        if(base::_has_exception(std::forward<Impl>(self)))
        {
          detail::_rethrow_exception<trait::is_exception_ptr_available<E>::value>{base::_exception<T, EC, E, exception_ptr_rethrow>(std::forward<Impl>(self))};
        }
        if(base::_has_error(std::forward<Impl>(self)))
        {
          detail::_rethrow_exception<trait::is_exception_ptr_available<EC>::value>{base::_error(std::forward<Impl>(self))};
        }
        OUTCOME_THROW_EXCEPTION(bad_outcome_access("no value")); // NOLINT
      }
    }
    template <class Impl> static constexpr void wide_error_check(Impl &&self)
    {
      if(!base::_has_error(std::forward<Impl>(self)))
      {
        OUTCOME_THROW_EXCEPTION(bad_outcome_access("no error")); // NOLINT
      }
    }
    template <class Impl> static constexpr void wide_exception_check(Impl &&self)
    {
      if(!base::_has_exception(std::forward<Impl>(self)))
      {
        OUTCOME_THROW_EXCEPTION(bad_outcome_access("no exception")); // NOLINT
      }
    }
  };
} // namespace policy
OUTCOME_V2_NAMESPACE_END
#endif
#endif
OUTCOME_V2_NAMESPACE_EXPORT_BEGIN
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class R, class S = std::error_code, class P = std::exception_ptr, class NoValuePolicy = policy::default_policy<R, S, P>> //
using outcome = std_outcome<R, S, P, NoValuePolicy>;
OUTCOME_V2_NAMESPACE_END
#endif
#include <iostream>
#include <sstream>
OUTCOME_V2_NAMESPACE_BEGIN
namespace detail
{
  template <class T> typename std::add_lvalue_reference<T>::type lvalueref() noexcept;
  template <template <class, class> class ValueStorage, class T, class E> inline std::ostream &value_storage_out(std::ostream &s, const ValueStorage<T, E> &v)
  {
    s << static_cast<uint16_t>(v._status.status_value) << " " << v._status.spare_storage_value << " ";
    if(v._status.have_value())
    {
      s << v._value; // NOLINT
    }
    if(v._status.have_error())
    {
      s << v._error; // NOLINT
    }
    return s;
  }
  template <template <class, class> class ValueStorage, class E> inline std::ostream &value_storage_out(std::ostream &s, const ValueStorage<void, E> &v)
  {
    s << static_cast<uint16_t>(v._status.status_value) << " " << v._status.spare_storage_value << " ";
    if(v._status.have_error())
    {
      s << v._error; // NOLINT
    }
    return s;
  }
  template <template <class, class> class ValueStorage, class T> inline std::ostream &value_storage_out(std::ostream &s, const ValueStorage<T, void> &v)
  {
    s << static_cast<uint16_t>(v._status.status_value) << " " << v._status.spare_storage_value << " ";
    if(v._status.have_value())
    {
      s << v._value; // NOLINT
    }
    return s;
  }
  template <class T, class E> inline std::ostream &operator<<(std::ostream &s, const value_storage_trivial<T, E> &v) { return value_storage_out(s, v); }
  template <class T, class E> inline std::ostream &operator<<(std::ostream &s, const value_storage_nontrivial<T, E> &v) { return value_storage_out(s, v); }
  template <template <class, class> class ValueStorage, class T, class E> inline std::istream &value_storage_in(std::istream &s, ValueStorage<T, E> &v)
  {
    using type = ValueStorage<T, E>;
    v.~type();
    new(&v) type;
    uint16_t x, y;
    s >> x >> y;
    v._status.status_value = static_cast<detail::status>(x);
    v._status.spare_storage_value = y;
    if(v._status.have_value())
    {
      new(&v._value) decltype(v._value)(); // NOLINT
      s >> v._value; // NOLINT
    }
    if(v._status.have_error())
    {
      new(&v._error) decltype(v._error)(); // NOLINT
      s >> v._error; // NOLINT
    }
    return s;
  }
  template <template <class, class> class ValueStorage, class E> inline std::istream &value_storage_in(std::istream &s, ValueStorage<void, E> &v)
  {
    using type = ValueStorage<void, E>;
    v.~type();
    new(&v) type;
    uint16_t x, y;
    s >> x >> y;
    v._status.status_value = static_cast<detail::status>(x);
    v._status.spare_storage_value = y;
    if(v._status.have_error())
    {
      new(&v._error) decltype(v._error)(); // NOLINT
      s >> v._error; // NOLINT
    }
    return s;
  }
  template <template <class, class> class ValueStorage, class T> inline std::istream &value_storage_in(std::istream &s, ValueStorage<T, void> &v)
  {
    using type = ValueStorage<T, void>;
    v.~type();
    new(&v) type;
    uint16_t x, y;
    s >> x >> y;
    v._status.status_value = static_cast<detail::status>(x);
    v._status.spare_storage_value = y;
    if(v._status.have_value())
    {
      new(&v._value) decltype(v._value)(); // NOLINT
      s >> v._value; // NOLINT
    }
    return s;
  }
  template <class T, class E> inline std::istream &operator>>(std::istream &s, value_storage_trivial<T, E> &v) { return value_storage_in(s, v); }
  template <class T, class E> inline std::istream &operator>>(std::istream &s, value_storage_nontrivial<T, E> &v) { return value_storage_in(s, v); }
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TPRED(!std::is_constructible<std::error_code, T>::value))
  inline std::string safe_message(T && /*unused*/) { return {}; }
  inline std::string safe_message(const std::error_code &ec) { return " (" + ec.message() + ")"; }
} // namespace detail
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
OUTCOME_TEMPLATE(class R, class S, class P)
OUTCOME_TREQUIRES(OUTCOME_TEXPR(detail::lvalueref<std::istream>() >> detail::lvalueref<R>()), OUTCOME_TEXPR(detail::lvalueref<std::istream>() >> detail::lvalueref<S>()))
inline std::istream &operator>>(std::istream &s, basic_result<R, S, P> &v)
{
  s >> v._iostreams_state();
  if(v.has_error())
  {
    s >> v.assume_error();
  }
  return s;
}
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
OUTCOME_TEMPLATE(class R, class S, class P)
OUTCOME_TREQUIRES(OUTCOME_TEXPR(detail::lvalueref<std::ostream>() << detail::lvalueref<R>()), OUTCOME_TEXPR(detail::lvalueref<std::ostream>() << detail::lvalueref<S>()))
inline std::ostream &operator<<(std::ostream &s, const basic_result<R, S, P> &v)
{
  s << v._iostreams_state();
  if(v.has_error())
  {
    s << v.assume_error();
  }
  return s;
}
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class R, class S, class P> inline std::string print(const basic_result<R, S, P> &v)
{
  std::stringstream s;
  if(v.has_value())
  {
    s << v.value();
  }
  if(v.has_error())
  {
    s << v.error() << detail::safe_message(v.error());
  }
  return s.str();
}
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class S, class P> inline std::string print(const basic_result<void, S, P> &v)
{
  std::stringstream s;
  if(v.has_value())
  {
    s << "(+void)";
  }
  if(v.has_error())
  {
    s << v.error() << detail::safe_message(v.error());
  }
  return s.str();
}
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class R, class P> inline std::string print(const basic_result<R, void, P> &v)
{
  std::stringstream s;
  if(v.has_value())
  {
    s << v.value();
  }
  if(v.has_error())
  {
    s << "(-void)";
  }
  return s.str();
}
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class P> inline std::string print(const basic_result<void, void, P> &v)
{
  std::stringstream s;
  if(v.has_value())
  {
    s << "(+void)";
  }
  if(v.has_error())
  {
    s << "(-void)";
  }
  return s.str();
}
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
OUTCOME_TEMPLATE(class R, class S, class P, class N)
OUTCOME_TREQUIRES(OUTCOME_TEXPR(detail::lvalueref<std::istream>() >> detail::lvalueref<R>()), OUTCOME_TEXPR(detail::lvalueref<std::istream>() >> detail::lvalueref<S>()), OUTCOME_TEXPR(detail::lvalueref<std::istream>() >> detail::lvalueref<P>()))
inline std::istream &operator>>(std::istream &s, outcome<R, S, P, N> &v)
{
  s >> v._iostreams_state();
  if(v.has_error())
  {
    s >> v.assume_error();
  }
  if(v.has_exception())
  {
    s >> v.assume_exception();
  }
  return s;
}
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
OUTCOME_TEMPLATE(class R, class S, class P, class N)
OUTCOME_TREQUIRES(OUTCOME_TEXPR(detail::lvalueref<std::ostream>() << detail::lvalueref<R>()), OUTCOME_TEXPR(detail::lvalueref<std::ostream>() << detail::lvalueref<S>()), OUTCOME_TEXPR(detail::lvalueref<std::ostream>() << detail::lvalueref<P>()))
inline std::ostream &operator<<(std::ostream &s, const outcome<R, S, P, N> &v)
{
  s << v._iostreams_state();
  if(v.has_error())
  {
    s << v.assume_error();
  }
  if(v.has_exception())
  {
    s << v.assume_exception();
  }
  return s;
}
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
template <class R, class S, class P, class N> inline std::string print(const outcome<R, S, P, N> &v)
{
  std::stringstream s;
  int total = static_cast<int>(v.has_value()) + static_cast<int>(v.has_error()) + static_cast<int>(v.has_exception());
  if(total > 1)
  {
    s << "{ ";
  }
  s << print(static_cast<const basic_result<R, S, N> &>(static_cast<const detail::basic_result_final<R, S, N> &>(v))); // NOLINT
  if(total > 1)
  {
    s << ", ";
  }
  if(v.has_exception())
  {
#ifdef __cpp_exceptions
    try
    {
      rethrow_exception(v.exception());
    }
    catch(const std::system_error &e)
    {
      s << "std::system_error code " << e.code() << ": " << e.what();
    }
    catch(const std::exception &e)
    {
      s << "std::exception: " << e.what();
    }
    catch(...)
#endif
    {
      s << "unknown exception";
    }
  }
  if(total > 1)
  {
    s << " }";
  }
  return s.str();
}
OUTCOME_V2_NAMESPACE_END
#endif
/* Try operation macros
(C) 2017-2021 Niall Douglas <http://www.nedproductions.biz/> (20 commits)
File Created: July 2017


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License in the accompanying file
Licence.txt or at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Distributed under the Boost Software License, Version 1.0.
    (See accompanying file Licence.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)
*/
#ifndef OUTCOME_TRY_HPP
#define OUTCOME_TRY_HPP
OUTCOME_V2_NAMESPACE_BEGIN
namespace detail
{
  struct has_value_overload
  {
  };
  struct as_failure_overload
  {
  };
  struct assume_error_overload
  {
  };
  struct error_overload
  {
  };
  struct assume_value_overload
  {
  };
  struct value_overload
  {
  };
  //#ifdef __APPLE__
  //  OUTCOME_TEMPLATE(class T, class R = decltype(std::declval<T>()._xcode_workaround_as_failure()))
  //#else
  OUTCOME_TEMPLATE(class T, class R = decltype(std::declval<T>().as_failure()))
  //#endif
  OUTCOME_TREQUIRES(OUTCOME_TPRED(OUTCOME_V2_NAMESPACE::is_failure_type<R>))
  constexpr inline bool has_as_failure(int /*unused */) { return true; }
  template <class T> constexpr inline bool has_as_failure(...) { return false; }
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<T>().assume_error()))
  constexpr inline bool has_assume_error(int /*unused */) { return true; }
  template <class T> constexpr inline bool has_assume_error(...) { return false; }
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<T>().error()))
  constexpr inline bool has_error(int /*unused */) { return true; }
  template <class T> constexpr inline bool has_error(...) { return false; }
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<T>().assume_value()))
  constexpr inline bool has_assume_value(int /*unused */) { return true; }
  template <class T> constexpr inline bool has_assume_value(...) { return false; }
  OUTCOME_TEMPLATE(class T)
  OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<T>().value()))
  constexpr inline bool has_value(int /*unused */) { return true; }
  template <class T> constexpr inline bool has_value(...) { return false; }
} // namespace detail
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
OUTCOME_TEMPLATE(class T)
OUTCOME_TREQUIRES(OUTCOME_TEXPR(std::declval<T>().has_value()))
constexpr inline bool try_operation_has_value(T &&v, detail::has_value_overload = {})
{
  return v.has_value();
}
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
OUTCOME_TEMPLATE(class T)
OUTCOME_TREQUIRES(OUTCOME_TPRED(detail::has_as_failure<T>(5)))
constexpr inline decltype(auto) try_operation_return_as(T &&v, detail::as_failure_overload = {})
{
  return static_cast<T &&>(v).as_failure();
}
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
OUTCOME_TEMPLATE(class T)
OUTCOME_TREQUIRES(OUTCOME_TPRED(!detail::has_as_failure<T>(5) && detail::has_assume_error<T>(5)))
constexpr inline decltype(auto) try_operation_return_as(T &&v, detail::assume_error_overload = {})
{
  return failure(static_cast<T &&>(v).assume_error());
}
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
OUTCOME_TEMPLATE(class T)
OUTCOME_TREQUIRES(OUTCOME_TPRED(!detail::has_as_failure<T>(5) && !detail::has_assume_error<T>(5) && detail::has_error<T>(5)))
constexpr inline decltype(auto) try_operation_return_as(T &&v, detail::error_overload = {})
{
  return failure(static_cast<T &&>(v).error());
}
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
OUTCOME_TEMPLATE(class T)
OUTCOME_TREQUIRES(OUTCOME_TPRED(detail::has_assume_value<T>(5)))
constexpr inline decltype(auto) try_operation_extract_value(T &&v, detail::assume_value_overload = {})
{
  return static_cast<T &&>(v).assume_value();
}
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
OUTCOME_TEMPLATE(class T)
OUTCOME_TREQUIRES(OUTCOME_TPRED(!detail::has_assume_value<T>(5) && detail::has_value<T>(5)))
constexpr inline decltype(auto) try_operation_extract_value(T &&v, detail::value_overload = {})
{
  return static_cast<T &&>(v).value();
}
OUTCOME_V2_NAMESPACE_END
#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 8
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wparentheses"
#endif
#define OUTCOME_TRY_GLUE2(x, y) x##y
#define OUTCOME_TRY_GLUE(x, y) OUTCOME_TRY_GLUE2(x, y)
#define OUTCOME_TRY_UNIQUE_NAME OUTCOME_TRY_GLUE(_outcome_try_unique_name_temporary, __COUNTER__)
#define OUTCOME_TRY_RETURN_ARG_COUNT(_1_, _2_, _3_, _4_, _5_, _6_, _7_, _8_, count, ...) count
#define OUTCOME_TRY_EXPAND_ARGS(args) OUTCOME_TRY_RETURN_ARG_COUNT args
#define OUTCOME_TRY_COUNT_ARGS_MAX8(...) OUTCOME_TRY_EXPAND_ARGS((__VA_ARGS__, 8, 7, 6, 5, 4, 3, 2, 1, 0))
#define OUTCOME_TRY_OVERLOAD_MACRO2(name, count) name##count
#define OUTCOME_TRY_OVERLOAD_MACRO1(name, count) OUTCOME_TRY_OVERLOAD_MACRO2(name, count)
#define OUTCOME_TRY_OVERLOAD_MACRO(name, count) OUTCOME_TRY_OVERLOAD_MACRO1(name, count)
#define OUTCOME_TRY_OVERLOAD_GLUE(x, y) x y
#define OUTCOME_TRY_CALL_OVERLOAD(name, ...) OUTCOME_TRY_OVERLOAD_GLUE(OUTCOME_TRY_OVERLOAD_MACRO(name, OUTCOME_TRY_COUNT_ARGS_MAX8(__VA_ARGS__)), (__VA_ARGS__))
#define _OUTCOME_TRY_RETURN_ARG_COUNT(_1_, _2_, _3_, _4_, _5_, _6_, _7_, _8_, count, ...) count
#define _OUTCOME_TRY_EXPAND_ARGS(args) _OUTCOME_TRY_RETURN_ARG_COUNT args
#define _OUTCOME_TRY_COUNT_ARGS_MAX8(...) _OUTCOME_TRY_EXPAND_ARGS((__VA_ARGS__, 8, 7, 6, 5, 4, 3, 2, 1, 0))
#define _OUTCOME_TRY_OVERLOAD_MACRO2(name, count) name##count
#define _OUTCOME_TRY_OVERLOAD_MACRO1(name, count) _OUTCOME_TRY_OVERLOAD_MACRO2(name, count)
#define _OUTCOME_TRY_OVERLOAD_MACRO(name, count) _OUTCOME_TRY_OVERLOAD_MACRO1(name, count)
#define _OUTCOME_TRY_OVERLOAD_GLUE(x, y) x y
#define _OUTCOME_TRY_CALL_OVERLOAD(name, ...) _OUTCOME_TRY_OVERLOAD_GLUE(_OUTCOME_TRY_OVERLOAD_MACRO(name, _OUTCOME_TRY_COUNT_ARGS_MAX8(__VA_ARGS__)), (__VA_ARGS__))
#ifndef OUTCOME_TRY_LIKELY_IF
#if (__cplusplus >= 202000 || _HAS_CXX20) && (!defined(__clang__) || __clang_major__ >= 12) && (!defined(__GNUC__) || defined(__clang__) || __GNUC__ >= 9)
#define OUTCOME_TRY_LIKELY_IF(...) if(__VA_ARGS__) [[likely]]
#elif defined(__clang__) || defined(__GNUC__)
#define OUTCOME_TRY_LIKELY_IF(...) if(__builtin_expect(!!(__VA_ARGS__), true))
#else
#define OUTCOME_TRY_LIKELY_IF(...) if(__VA_ARGS__)
#endif
#endif
#define OUTCOME_TRYV2_UNIQUE_STORAGE_UNPACK(...) __VA_ARGS__
#define OUTCOME_TRYV2_UNIQUE_STORAGE_DEDUCE3(unique, ...) auto unique = (__VA_ARGS__)
#define OUTCOME_TRYV2_UNIQUE_STORAGE_DEDUCE2(x) x
#define OUTCOME_TRYV2_UNIQUE_STORAGE_DEDUCE(unique, x, ...) OUTCOME_TRYV2_UNIQUE_STORAGE_DEDUCE2(OUTCOME_TRYV2_UNIQUE_STORAGE_DEDUCE3(unique, __VA_ARGS__))
#define OUTCOME_TRYV2_UNIQUE_STORAGE_SPECIFIED3(unique, x, y, ...) x unique = (__VA_ARGS__)
#define OUTCOME_TRYV2_UNIQUE_STORAGE_SPECIFIED2(x) x
#define OUTCOME_TRYV2_UNIQUE_STORAGE_SPECIFIED(unique, ...) OUTCOME_TRYV2_UNIQUE_STORAGE_SPECIFIED2(OUTCOME_TRYV2_UNIQUE_STORAGE_SPECIFIED3(unique, __VA_ARGS__))
#define OUTCOME_TRYV2_UNIQUE_STORAGE1(...) OUTCOME_TRYV2_UNIQUE_STORAGE_DEDUCE
#define OUTCOME_TRYV2_UNIQUE_STORAGE2(...) OUTCOME_TRYV2_UNIQUE_STORAGE_SPECIFIED
#define OUTCOME_TRYV2_UNIQUE_STORAGE(unique, spec, ...) _OUTCOME_TRY_CALL_OVERLOAD(OUTCOME_TRYV2_UNIQUE_STORAGE, OUTCOME_TRYV2_UNIQUE_STORAGE_UNPACK spec) (unique, OUTCOME_TRYV2_UNIQUE_STORAGE_UNPACK spec, __VA_ARGS__)
// Use if(!expr); else as some compilers assume else clauses are always unlikely
#define OUTCOME_TRYV2_SUCCESS_LIKELY(unique, retstmt, spec, ...) OUTCOME_TRYV2_UNIQUE_STORAGE(unique, spec, __VA_ARGS__); OUTCOME_TRY_LIKELY_IF(::OUTCOME_V2_NAMESPACE::try_operation_has_value(unique)); else retstmt ::OUTCOME_V2_NAMESPACE::try_operation_return_as(static_cast<decltype(unique) &&>(unique))
#define OUTCOME_TRYV3_FAILURE_LIKELY(unique, retstmt, spec, ...) OUTCOME_TRYV2_UNIQUE_STORAGE(unique, spec, __VA_ARGS__); OUTCOME_TRY_LIKELY_IF(!OUTCOME_V2_NAMESPACE::try_operation_has_value(unique)) retstmt ::OUTCOME_V2_NAMESPACE::try_operation_return_as(static_cast<decltype(unique) &&>(unique))
#define OUTCOME_TRY2_VAR_SECOND2(x, var) var
#define OUTCOME_TRY2_VAR_SECOND3(x, y, ...) x y
#define OUTCOME_TRY2_VAR(spec) _OUTCOME_TRY_CALL_OVERLOAD(OUTCOME_TRY2_VAR_SECOND, OUTCOME_TRYV2_UNIQUE_STORAGE_UNPACK spec, spec)
#define OUTCOME_TRY2_SUCCESS_LIKELY(unique, retstmt, var, ...) OUTCOME_TRYV2_SUCCESS_LIKELY(unique, retstmt, var, __VA_ARGS__); OUTCOME_TRY2_VAR(var) = ::OUTCOME_V2_NAMESPACE::try_operation_extract_value(static_cast<decltype(unique) &&>(unique))
#define OUTCOME_TRY2_FAILURE_LIKELY(unique, retstmt, var, ...) OUTCOME_TRYV3_FAILURE_LIKELY(unique, retstmt, var, __VA_ARGS__); OUTCOME_TRY2_VAR(var) = ::OUTCOME_V2_NAMESPACE::try_operation_extract_value(static_cast<decltype(unique) &&>(unique))
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_TRYV(...) OUTCOME_TRYV2_SUCCESS_LIKELY(OUTCOME_TRY_UNIQUE_NAME, return, deduce, __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_TRYV_FAILURE_LIKELY(...) OUTCOME_TRYV3_FAILURE_LIKELY(OUTCOME_TRY_UNIQUE_NAME, return, deduce, __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_CO_TRYV(...) OUTCOME_TRYV2_SUCCESS_LIKELY(OUTCOME_TRY_UNIQUE_NAME, co_return, deduce, __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_CO_TRYV_FAILURE_LIKELY(...) OUTCOME_TRYV3_FAILURE_LIKELY(OUTCOME_TRY_UNIQUE_NAME, co_return, deduce, __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_TRYV2(s, ...) OUTCOME_TRYV2_SUCCESS_LIKELY(OUTCOME_TRY_UNIQUE_NAME, return, (s,), __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_TRYV2_FAILURE_LIKELY(s, ...) OUTCOME_TRYV3_FAILURE_LIKELY(OUTCOME_TRY_UNIQUE_NAME, return, (s,), __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_CO_TRYV2(s, ...) OUTCOME_TRYV2_SUCCESS_LIKELY(OUTCOME_TRY_UNIQUE_NAME, co_return, (s,), __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_CO_TRYV2_FAILURE_LIKELY(s, ...) OUTCOME_TRYV3_FAILURE_LIKELY(OUTCOME_TRY_UNIQUE_NAME, co_return, s(,), __VA_ARGS__)
#if defined(__GNUC__) || defined(__clang__)
#define OUTCOME_TRYX2(unique, retstmt, ...) ({ OUTCOME_TRYV2_SUCCESS_LIKELY(unique, retstmt, deduce, __VA_ARGS__); ::OUTCOME_V2_NAMESPACE::try_operation_extract_value(static_cast<decltype(unique) &&>(unique)); })
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_TRYX(...) OUTCOME_TRYX2(OUTCOME_TRY_UNIQUE_NAME, return, __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_CO_TRYX(...) OUTCOME_TRYX2(OUTCOME_TRY_UNIQUE_NAME, co_return, __VA_ARGS__)
#endif
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_TRYA(v, ...) OUTCOME_TRY2_SUCCESS_LIKELY(OUTCOME_TRY_UNIQUE_NAME, return, v, __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_TRYA_FAILURE_LIKELY(v, ...) OUTCOME_TRY2_FAILURE_LIKELY(OUTCOME_TRY_UNIQUE_NAME, return, v, __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_CO_TRYA(v, ...) OUTCOME_TRY2_SUCCESS_LIKELY(OUTCOME_TRY_UNIQUE_NAME, co_return, v, __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_CO_TRYA_FAILURE_LIKELY(v, ...) OUTCOME_TRY2_FAILURE_LIKELY(OUTCOME_TRY_UNIQUE_NAME, co_return, v, __VA_ARGS__)
#define OUTCOME_TRY_INVOKE_TRY8(a, b, c, d, e, f, g, h) OUTCOME_TRYA(a, b, c, d, e, f, g, h)
#define OUTCOME_TRY_INVOKE_TRY7(a, b, c, d, e, f, g) OUTCOME_TRYA(a, b, c, d, e, f, g)
#define OUTCOME_TRY_INVOKE_TRY6(a, b, c, d, e, f) OUTCOME_TRYA(a, b, c, d, e, f)
#define OUTCOME_TRY_INVOKE_TRY5(a, b, c, d, e) OUTCOME_TRYA(a, b, c, d, e)
#define OUTCOME_TRY_INVOKE_TRY4(a, b, c, d) OUTCOME_TRYA(a, b, c, d)
#define OUTCOME_TRY_INVOKE_TRY3(a, b, c) OUTCOME_TRYA(a, b, c)
#define OUTCOME_TRY_INVOKE_TRY2(a, b) OUTCOME_TRYA(a, b)
#define OUTCOME_TRY_INVOKE_TRY1(a) OUTCOME_TRYV(a)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_TRY(...) OUTCOME_TRY_CALL_OVERLOAD(OUTCOME_TRY_INVOKE_TRY, __VA_ARGS__)
#define OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY8(a, b, c, d, e, f, g, h) OUTCOME_TRYA_FAILURE_LIKELY(a, b, c, d, e, f, g, h)
#define OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY7(a, b, c, d, e, f, g) OUTCOME_TRYA_FAILURE_LIKELY(a, b, c, d, e, f, g)
#define OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY6(a, b, c, d, e, f) OUTCOME_TRYA_FAILURE_LIKELY(a, b, c, d, e, f)
#define OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY5(a, b, c, d, e) OUTCOME_TRYA_FAILURE_LIKELY(a, b, c, d, e)
#define OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY4(a, b, c, d) OUTCOME_TRYA_FAILURE_LIKELY(a, b, c, d)
#define OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY3(a, b, c) OUTCOME_TRYA_FAILURE_LIKELY(a, b, c)
#define OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY2(a, b) OUTCOME_TRYA_FAILURE_LIKELY(a, b)
#define OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY1(a) OUTCOME_TRYV_FAILURE_LIKELY(a)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_TRY_FAILURE_LIKELY(...) OUTCOME_TRY_CALL_OVERLOAD(OUTCOME_TRY_FAILURE_LIKELY_INVOKE_TRY, __VA_ARGS__)
#define OUTCOME_CO_TRY_INVOKE_TRY8(a, b, c, d, e, f, g, h) OUTCOME_CO_TRYA(a, b, c, d, e, f, g, h)
#define OUTCOME_CO_TRY_INVOKE_TRY7(a, b, c, d, e, f, g) OUTCOME_CO_TRYA(a, b, c, d, e, f, g)
#define OUTCOME_CO_TRY_INVOKE_TRY6(a, b, c, d, e, f) OUTCOME_CO_TRYA(a, b, c, d, e, f)
#define OUTCOME_CO_TRY_INVOKE_TRY5(a, b, c, d, e) OUTCOME_CO_TRYA(a, b, c, d, e)
#define OUTCOME_CO_TRY_INVOKE_TRY4(a, b, c, d) OUTCOME_CO_TRYA(a, b, c, d)
#define OUTCOME_CO_TRY_INVOKE_TRY3(a, b, c) OUTCOME_CO_TRYA(a, b, c)
#define OUTCOME_CO_TRY_INVOKE_TRY2(a, b) OUTCOME_CO_TRYA(a, b)
#define OUTCOME_CO_TRY_INVOKE_TRY1(a) OUTCOME_CO_TRYV(a)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_CO_TRY(...) OUTCOME_TRY_CALL_OVERLOAD(OUTCOME_CO_TRY_INVOKE_TRY, __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_TRY(...) OUTCOME_TRY_CALL_OVERLOAD(OUTCOME_TRY_INVOKE_TRY, __VA_ARGS__)
#define OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY8(a, b, c, d, e, f, g, h) OUTCOME_CO_TRYA_FAILURE_LIKELY(a, b, c, d, e, f, g, h)
#define OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY7(a, b, c, d, e, f, g) OUTCOME_CO_TRYA_FAILURE_LIKELY(a, b, c, d, e, f, g)
#define OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY6(a, b, c, d, e, f) OUTCOME_CO_TRYA_FAILURE_LIKELY(a, b, c, d, e, f)
#define OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY5(a, b, c, d, e) OUTCOME_CO_TRYA_FAILURE_LIKELY(a, b, c, d, e)
#define OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY4(a, b, c, d) OUTCOME_CO_TRYA_FAILURE_LIKELY(a, b, c, d)
#define OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY3(a, b, c) OUTCOME_CO_TRYA_FAILURE_LIKELY(a, b, c)
#define OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY2(a, b) OUTCOME_CO_TRYA_FAILURE_LIKELY(a, b)
#define OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY1(a) OUTCOME_CO_TRYV_FAILURE_LIKELY(a)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME_CO_TRY_FAILURE_LIKELY(...) OUTCOME_TRY_CALL_OVERLOAD(OUTCOME_CO_TRY_FAILURE_LIKELY_INVOKE_TRY, __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME21_TRYA(v, ...) OUTCOME_TRY2_SUCCESS_LIKELY(OUTCOME_TRY_UNIQUE_NAME, return, deduce, auto &&v, __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME21_TRYA_FAILURE_LIKELY(v, ...) OUTCOME_TRY2_FAILURE_LIKELY(OUTCOME_TRY_UNIQUE_NAME, return, deduce, auto &&v, __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME21_CO_TRYA(v, ...) OUTCOME_TRY2_FAILURE_LIKELY(OUTCOME_TRY_UNIQUE_NAME, co_return, deduce, auto &&v, __VA_ARGS__)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME21_CO_TRYA_FAILURE_LIKELY(v, ...) OUTCOME_TRY2_FAILURE_LIKELY(OUTCOME_TRY_UNIQUE_NAME, co_retrn, deduce, auto &&v, __VA_ARGS__)
#define OUTCOME21_TRY_INVOKE_TRY8(a, b, c, d, e, f, g, h) OUTCOME21_TRYA(a, b, c, d, e, f, g, h)
#define OUTCOME21_TRY_INVOKE_TRY7(a, b, c, d, e, f, g) OUTCOME21_TRYA(a, b, c, d, e, f, g)
#define OUTCOME21_TRY_INVOKE_TRY6(a, b, c, d, e, f) OUTCOME21_TRYA(a, b, c, d, e, f)
#define OUTCOME21_TRY_INVOKE_TRY5(a, b, c, d, e) OUTCOME21_TRYA(a, b, c, d, e)
#define OUTCOME21_TRY_INVOKE_TRY4(a, b, c, d) OUTCOME21_TRYA(a, b, c, d)
#define OUTCOME21_TRY_INVOKE_TRY3(a, b, c) OUTCOME21_TRYA(a, b, c)
#define OUTCOME21_TRY_INVOKE_TRY2(a, b) OUTCOME21_TRYA(a, b)
#define OUTCOME21_TRY_INVOKE_TRY1(a) OUTCOME_TRYV(a)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME21_TRY(...) OUTCOME_TRY_CALL_OVERLOAD(OUTCOME21_TRY_INVOKE_TRY, __VA_ARGS__)
#define OUTCOME21_TRY_FAILURE_LIKELY_INVOKE_TRY8(a, b, c, d, e, f, g, h) OUTCOME21_TRYA_FAILURE_LIKELY(a, b, c, d, e, f, g, h)
#define OUTCOME21_TRY_FAILURE_LIKELY_INVOKE_TRY7(a, b, c, d, e, f, g) OUTCOME21_TRYA_FAILURE_LIKELY(a, b, c, d, e, f, g)
#define OUTCOME21_TRY_FAILURE_LIKELY_INVOKE_TRY6(a, b, c, d, e, f) OUTCOME21_TRYA_FAILURE_LIKELY(a, b, c, d, e, f)
#define OUTCOME21_TRY_FAILURE_LIKELY_INVOKE_TRY5(a, b, c, d, e) OUTCOME21_TRYA_FAILURE_LIKELY(a, b, c, d, e)
#define OUTCOME21_TRY_FAILURE_LIKELY_INVOKE_TRY4(a, b, c, d) OUTCOME21_TRYA_FAILURE_LIKELY(a, b, c, d)
#define OUTCOME21_TRY_FAILURE_LIKELY_INVOKE_TRY3(a, b, c) OUTCOME21_TRYA_FAILURE_LIKELY(a, b, c)
#define OUTCOME21_TRY_FAILURE_LIKELY_INVOKE_TRY2(a, b) OUTCOME21_TRYA_FAILURE_LIKELY(a, b)
#define OUTCOME21_TRY_FAILURE_LIKELY_INVOKE_TRY1(a) OUTCOME_TRYV_FAILURE_LIKELY(a)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME21_TRY_FAILURE_LIKELY(...) OUTCOME_TRY_CALL_OVERLOAD(OUTCOME21_TRY_FAILURE_LIKELY_INVOKE_TRY, __VA_ARGS__)
#define OUTCOME21_CO_TRY_INVOKE_TRY8(a, b, c, d, e, f, g, h) OUTCOME21_CO_TRYA(a, b, c, d, e, f, g, h)
#define OUTCOME21_CO_TRY_INVOKE_TRY7(a, b, c, d, e, f, g) OUTCOME21_CO_TRYA(a, b, c, d, e, f, g)
#define OUTCOME21_CO_TRY_INVOKE_TRY6(a, b, c, d, e, f) OUTCOME21_CO_TRYA(a, b, c, d, e, f)
#define OUTCOME21_CO_TRY_INVOKE_TRY5(a, b, c, d, e) OUTCOME21_CO_TRYA(a, b, c, d, e)
#define OUTCOME21_CO_TRY_INVOKE_TRY4(a, b, c, d) OUTCOME21_CO_TRYA(a, b, c, d)
#define OUTCOME21_CO_TRY_INVOKE_TRY3(a, b, c) OUTCOME21_CO_TRYA(a, b, c)
#define OUTCOME21_CO_TRY_INVOKE_TRY2(a, b) OUTCOME21_CO_TRYA(a, b)
#define OUTCOME21_CO_TRY_INVOKE_TRY1(a) OUTCOME_CO_TRYV(a)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME21_CO_TRY(...) OUTCOME_TRY_CALL_OVERLOAD(OUTCOME21_CO_TRY_INVOKE_TRY, __VA_ARGS__)
#define OUTCOME21_CO_TRY_FAILURE_LIKELY_INVOKE_TRY8(a, b, c, d, e, f, g, h) OUTCOME21_CO_TRYA_FAILURE_LIKELY(a, b, c, d, e, f, g, h)
#define OUTCOME21_CO_TRY_FAILURE_LIKELY_INVOKE_TRY7(a, b, c, d, e, f, g) OUTCOME21_CO_TRYA_FAILURE_LIKELY(a, b, c, d, e, f, g)
#define OUTCOME21_CO_TRY_FAILURE_LIKELY_INVOKE_TRY6(a, b, c, d, e, f) OUTCOME21_CO_TRYA_FAILURE_LIKELY(a, b, c, d, e, f)
#define OUTCOME21_CO_TRY_FAILURE_LIKELY_INVOKE_TRY5(a, b, c, d, e) OUTCOME21_CO_TRYA_FAILURE_LIKELY(a, b, c, d, e)
#define OUTCOME21_CO_TRY_FAILURE_LIKELY_INVOKE_TRY4(a, b, c, d) OUTCOME21_CO_TRYA_FAILURE_LIKELY(a, b, c, d)
#define OUTCOME21_CO_TRY_FAILURE_LIKELY_INVOKE_TRY3(a, b, c) OUTCOME21_CO_TRYA_FAILURE_LIKELY(a, b, c)
#define OUTCOME21_CO_TRY_FAILURE_LIKELY_INVOKE_TRY2(a, b) OUTCOME21_CO_TRYA_FAILURE_LIKELY(a, b)
#define OUTCOME21_CO_TRY_FAILURE_LIKELY_INVOKE_TRY1(a) OUTCOME_CO_TRYV_FAILURE_LIKELY(a)
/*! AWAITING HUGO JSON CONVERSION TOOL
SIGNATURE NOT RECOGNISED
*/
#define OUTCOME21_CO_TRY_FAILURE_LIKELY(...) OUTCOME_TRY_CALL_OVERLOAD(OUTCOME21_CO_TRY_FAILURE_LIKELY_INVOKE_TRY, __VA_ARGS__)
#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 8
#pragma GCC diagnostic pop
#endif
#endif
#else
import OUTCOME_V2_CXX_MODULE_NAME;
#endif
