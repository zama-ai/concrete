// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

// generated: see genDynamicRandAndArityCall.py

#ifndef CONCRETELANG_SERVERLIB_DYNAMIC_ARITY_CALL_H
#define CONCRETELANG_SERVERLIB_DYNAMIC_ARITY_CALL_H

#include <cassert>
#include <vector>

#include "concretelang/ClientLib/Types.h"

namespace concretelang {
namespace serverlib {

template <typename Res>
Res multi_arity_call(Res (*func)(void *...), std::vector<void *> args) {
  switch (args.size()) {
    // TODO C17++: https://en.cppreference.com/w/cpp/utility/apply
  case 1:
    return func(args[0]);
  case 2:
    return func(args[0], args[1]);
  case 3:
    return func(args[0], args[1], args[2]);
  case 4:
    return func(args[0], args[1], args[2], args[3]);
  case 5:
    return func(args[0], args[1], args[2], args[3], args[4]);
  case 6:
    return func(args[0], args[1], args[2], args[3], args[4], args[5]);
  case 7:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6]);
  case 8:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7]);
  case 9:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8]);
  case 10:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9]);
  case 11:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10]);
  case 12:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11]);
  case 13:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12]);
  case 14:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13]);
  case 15:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14]);
  case 16:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15]);
  case 17:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16]);
  case 18:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17]);
  case 19:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18]);
  case 20:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19]);
  case 21:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20]);
  case 22:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21]);
  case 23:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22]);
  case 24:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23]);
  case 25:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24]);
  case 26:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25]);
  case 27:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26]);
  case 28:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27]);
  case 29:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28]);
  case 30:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29]);
  case 31:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30]);
  case 32:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31]);
  case 33:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32]);
  case 34:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33]);
  case 35:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34]);
  case 36:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35]);
  case 37:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36]);
  case 38:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37]);
  case 39:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37], args[38]);
  case 40:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37], args[38], args[39]);
  case 41:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37], args[38], args[39], args[40]);
  case 42:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37], args[38], args[39], args[40], args[41]);
  case 43:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37], args[38], args[39], args[40], args[41], args[42]);
  case 44:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37], args[38], args[39], args[40], args[41], args[42],
                args[43]);
  case 45:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37], args[38], args[39], args[40], args[41], args[42],
                args[43], args[44]);
  case 46:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37], args[38], args[39], args[40], args[41], args[42],
                args[43], args[44], args[45]);
  case 47:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37], args[38], args[39], args[40], args[41], args[42],
                args[43], args[44], args[45], args[46]);
  case 48:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37], args[38], args[39], args[40], args[41], args[42],
                args[43], args[44], args[45], args[46], args[47]);
  case 49:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37], args[38], args[39], args[40], args[41], args[42],
                args[43], args[44], args[45], args[46], args[47], args[48]);
  case 50:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49]);
  case 51:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37], args[38], args[39], args[40], args[41], args[42],
                args[43], args[44], args[45], args[46], args[47], args[48],
                args[49], args[50]);
  case 52:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37], args[38], args[39], args[40], args[41], args[42],
                args[43], args[44], args[45], args[46], args[47], args[48],
                args[49], args[50], args[51]);
  case 53:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37], args[38], args[39], args[40], args[41], args[42],
                args[43], args[44], args[45], args[46], args[47], args[48],
                args[49], args[50], args[51], args[52]);
  case 54:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37], args[38], args[39], args[40], args[41], args[42],
                args[43], args[44], args[45], args[46], args[47], args[48],
                args[49], args[50], args[51], args[52], args[53]);
  case 55:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37], args[38], args[39], args[40], args[41], args[42],
                args[43], args[44], args[45], args[46], args[47], args[48],
                args[49], args[50], args[51], args[52], args[53], args[54]);
  case 56:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55]);
  case 57:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56]);
  case 58:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37], args[38], args[39], args[40], args[41], args[42],
                args[43], args[44], args[45], args[46], args[47], args[48],
                args[49], args[50], args[51], args[52], args[53], args[54],
                args[55], args[56], args[57]);
  case 59:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37], args[38], args[39], args[40], args[41], args[42],
                args[43], args[44], args[45], args[46], args[47], args[48],
                args[49], args[50], args[51], args[52], args[53], args[54],
                args[55], args[56], args[57], args[58]);
  case 60:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37], args[38], args[39], args[40], args[41], args[42],
                args[43], args[44], args[45], args[46], args[47], args[48],
                args[49], args[50], args[51], args[52], args[53], args[54],
                args[55], args[56], args[57], args[58], args[59]);
  case 61:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37], args[38], args[39], args[40], args[41], args[42],
                args[43], args[44], args[45], args[46], args[47], args[48],
                args[49], args[50], args[51], args[52], args[53], args[54],
                args[55], args[56], args[57], args[58], args[59], args[60]);
  case 62:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61]);
  case 63:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62]);
  case 64:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63]);
  case 65:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37], args[38], args[39], args[40], args[41], args[42],
                args[43], args[44], args[45], args[46], args[47], args[48],
                args[49], args[50], args[51], args[52], args[53], args[54],
                args[55], args[56], args[57], args[58], args[59], args[60],
                args[61], args[62], args[63], args[64]);
  case 66:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37], args[38], args[39], args[40], args[41], args[42],
                args[43], args[44], args[45], args[46], args[47], args[48],
                args[49], args[50], args[51], args[52], args[53], args[54],
                args[55], args[56], args[57], args[58], args[59], args[60],
                args[61], args[62], args[63], args[64], args[65]);
  case 67:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37], args[38], args[39], args[40], args[41], args[42],
                args[43], args[44], args[45], args[46], args[47], args[48],
                args[49], args[50], args[51], args[52], args[53], args[54],
                args[55], args[56], args[57], args[58], args[59], args[60],
                args[61], args[62], args[63], args[64], args[65], args[66]);
  case 68:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67]);
  case 69:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68]);
  case 70:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69]);
  case 71:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70]);
  case 72:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37], args[38], args[39], args[40], args[41], args[42],
                args[43], args[44], args[45], args[46], args[47], args[48],
                args[49], args[50], args[51], args[52], args[53], args[54],
                args[55], args[56], args[57], args[58], args[59], args[60],
                args[61], args[62], args[63], args[64], args[65], args[66],
                args[67], args[68], args[69], args[70], args[71]);
  case 73:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37], args[38], args[39], args[40], args[41], args[42],
                args[43], args[44], args[45], args[46], args[47], args[48],
                args[49], args[50], args[51], args[52], args[53], args[54],
                args[55], args[56], args[57], args[58], args[59], args[60],
                args[61], args[62], args[63], args[64], args[65], args[66],
                args[67], args[68], args[69], args[70], args[71], args[72]);
  case 74:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73]);
  case 75:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74]);
  case 76:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75]);
  case 77:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76]);
  case 78:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77]);
  case 79:
    return func(args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                args[7], args[8], args[9], args[10], args[11], args[12],
                args[13], args[14], args[15], args[16], args[17], args[18],
                args[19], args[20], args[21], args[22], args[23], args[24],
                args[25], args[26], args[27], args[28], args[29], args[30],
                args[31], args[32], args[33], args[34], args[35], args[36],
                args[37], args[38], args[39], args[40], args[41], args[42],
                args[43], args[44], args[45], args[46], args[47], args[48],
                args[49], args[50], args[51], args[52], args[53], args[54],
                args[55], args[56], args[57], args[58], args[59], args[60],
                args[61], args[62], args[63], args[64], args[65], args[66],
                args[67], args[68], args[69], args[70], args[71], args[72],
                args[73], args[74], args[75], args[76], args[77], args[78]);
  case 80:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79]);
  case 81:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80]);
  case 82:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81]);
  case 83:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82]);
  case 84:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83]);
  case 85:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84]);
  case 86:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85]);
  case 87:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86]);
  case 88:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87]);
  case 89:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88]);
  case 90:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89]);
  case 91:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90]);
  case 92:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91]);
  case 93:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92]);
  case 94:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93]);
  case 95:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94]);
  case 96:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95]);
  case 97:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96]);
  case 98:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97]);
  case 99:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98]);
  case 100:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99]);
  case 101:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100]);
  case 102:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100], args[101]);
  case 103:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100], args[101], args[102]);
  case 104:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100], args[101], args[102], args[103]);
  case 105:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100], args[101], args[102], args[103], args[104]);
  case 106:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100], args[101], args[102], args[103], args[104],
        args[105]);
  case 107:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100], args[101], args[102], args[103], args[104],
        args[105], args[106]);
  case 108:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100], args[101], args[102], args[103], args[104],
        args[105], args[106], args[107]);
  case 109:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100], args[101], args[102], args[103], args[104],
        args[105], args[106], args[107], args[108]);
  case 110:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100], args[101], args[102], args[103], args[104],
        args[105], args[106], args[107], args[108], args[109]);
  case 111:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100], args[101], args[102], args[103], args[104],
        args[105], args[106], args[107], args[108], args[109], args[110]);
  case 112:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100], args[101], args[102], args[103], args[104],
        args[105], args[106], args[107], args[108], args[109], args[110],
        args[111]);
  case 113:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100], args[101], args[102], args[103], args[104],
        args[105], args[106], args[107], args[108], args[109], args[110],
        args[111], args[112]);
  case 114:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100], args[101], args[102], args[103], args[104],
        args[105], args[106], args[107], args[108], args[109], args[110],
        args[111], args[112], args[113]);
  case 115:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100], args[101], args[102], args[103], args[104],
        args[105], args[106], args[107], args[108], args[109], args[110],
        args[111], args[112], args[113], args[114]);
  case 116:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100], args[101], args[102], args[103], args[104],
        args[105], args[106], args[107], args[108], args[109], args[110],
        args[111], args[112], args[113], args[114], args[115]);
  case 117:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100], args[101], args[102], args[103], args[104],
        args[105], args[106], args[107], args[108], args[109], args[110],
        args[111], args[112], args[113], args[114], args[115], args[116]);
  case 118:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100], args[101], args[102], args[103], args[104],
        args[105], args[106], args[107], args[108], args[109], args[110],
        args[111], args[112], args[113], args[114], args[115], args[116],
        args[117]);
  case 119:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100], args[101], args[102], args[103], args[104],
        args[105], args[106], args[107], args[108], args[109], args[110],
        args[111], args[112], args[113], args[114], args[115], args[116],
        args[117], args[118]);
  case 120:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100], args[101], args[102], args[103], args[104],
        args[105], args[106], args[107], args[108], args[109], args[110],
        args[111], args[112], args[113], args[114], args[115], args[116],
        args[117], args[118], args[119]);
  case 121:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100], args[101], args[102], args[103], args[104],
        args[105], args[106], args[107], args[108], args[109], args[110],
        args[111], args[112], args[113], args[114], args[115], args[116],
        args[117], args[118], args[119], args[120]);
  case 122:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100], args[101], args[102], args[103], args[104],
        args[105], args[106], args[107], args[108], args[109], args[110],
        args[111], args[112], args[113], args[114], args[115], args[116],
        args[117], args[118], args[119], args[120], args[121]);
  case 123:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100], args[101], args[102], args[103], args[104],
        args[105], args[106], args[107], args[108], args[109], args[110],
        args[111], args[112], args[113], args[114], args[115], args[116],
        args[117], args[118], args[119], args[120], args[121], args[122]);
  case 124:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100], args[101], args[102], args[103], args[104],
        args[105], args[106], args[107], args[108], args[109], args[110],
        args[111], args[112], args[113], args[114], args[115], args[116],
        args[117], args[118], args[119], args[120], args[121], args[122],
        args[123]);
  case 125:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100], args[101], args[102], args[103], args[104],
        args[105], args[106], args[107], args[108], args[109], args[110],
        args[111], args[112], args[113], args[114], args[115], args[116],
        args[117], args[118], args[119], args[120], args[121], args[122],
        args[123], args[124]);
  case 126:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100], args[101], args[102], args[103], args[104],
        args[105], args[106], args[107], args[108], args[109], args[110],
        args[111], args[112], args[113], args[114], args[115], args[116],
        args[117], args[118], args[119], args[120], args[121], args[122],
        args[123], args[124], args[125]);
  case 127:
    return func(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
        args[8], args[9], args[10], args[11], args[12], args[13], args[14],
        args[15], args[16], args[17], args[18], args[19], args[20], args[21],
        args[22], args[23], args[24], args[25], args[26], args[27], args[28],
        args[29], args[30], args[31], args[32], args[33], args[34], args[35],
        args[36], args[37], args[38], args[39], args[40], args[41], args[42],
        args[43], args[44], args[45], args[46], args[47], args[48], args[49],
        args[50], args[51], args[52], args[53], args[54], args[55], args[56],
        args[57], args[58], args[59], args[60], args[61], args[62], args[63],
        args[64], args[65], args[66], args[67], args[68], args[69], args[70],
        args[71], args[72], args[73], args[74], args[75], args[76], args[77],
        args[78], args[79], args[80], args[81], args[82], args[83], args[84],
        args[85], args[86], args[87], args[88], args[89], args[90], args[91],
        args[92], args[93], args[94], args[95], args[96], args[97], args[98],
        args[99], args[100], args[101], args[102], args[103], args[104],
        args[105], args[106], args[107], args[108], args[109], args[110],
        args[111], args[112], args[113], args[114], args[115], args[116],
        args[117], args[118], args[119], args[120], args[121], args[122],
        args[123], args[124], args[125], args[126]);

  default:
    assert(false);
  }
}

} // namespace serverlib
} // namespace concretelang

#endif
