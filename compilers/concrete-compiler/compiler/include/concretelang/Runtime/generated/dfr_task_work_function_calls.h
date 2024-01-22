case 1: {
  void *output1;
  _dfr_checked_aligned_alloc(&output1, 512, inputs.output_sizes[0]);
  switch (inputs.params.size()) {
  case 0:
    wfn(output1);
    break;
  case 1:
    wfn(output1, inputs.params[0]);
    break;
  case 2:
    wfn(output1, inputs.params[0], inputs.params[1]);
    break;
  case 3:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2]);
    break;
  case 4:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3]);
    break;
  case 5:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4]);
    break;
  case 6:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5]);
    break;
  case 7:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6]);
    break;
  case 8:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7]);
    break;
  case 9:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8]);
    break;
  case 10:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9]);
    break;
  case 11:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10]);
    break;
  case 12:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11]);
    break;
  case 13:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12]);
    break;
  case 14:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13]);
    break;
  case 15:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14]);
    break;
  case 16:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15]);
    break;
  case 17:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16]);
    break;
  case 18:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17]);
    break;
  case 19:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18]);
    break;
  case 20:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19]);
    break;
  case 21:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20]);
    break;
  case 22:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21]);
    break;
  case 23:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22]);
    break;
  case 24:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23]);
    break;
  case 25:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24]);
    break;
  case 26:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25]);
    break;
  case 27:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26]);
    break;
  case 28:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27]);
    break;
  case 29:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28]);
    break;
  case 30:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29]);
    break;
  case 31:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30]);
    break;
  case 32:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31]);
    break;
  case 33:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32]);
    break;
  case 34:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33]);
    break;
  case 35:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34]);
    break;
  case 36:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35]);
    break;
  case 37:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36]);
    break;
  case 38:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36], inputs.params[37]);
    break;
  case 39:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36], inputs.params[37],
        inputs.params[38]);
    break;
  case 40:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36], inputs.params[37],
        inputs.params[38], inputs.params[39]);
    break;
  case 41:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36], inputs.params[37],
        inputs.params[38], inputs.params[39], inputs.params[40]);
    break;
  case 42:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36], inputs.params[37],
        inputs.params[38], inputs.params[39], inputs.params[40],
        inputs.params[41]);
    break;
  case 43:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36], inputs.params[37],
        inputs.params[38], inputs.params[39], inputs.params[40],
        inputs.params[41], inputs.params[42]);
    break;
  case 44:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36], inputs.params[37],
        inputs.params[38], inputs.params[39], inputs.params[40],
        inputs.params[41], inputs.params[42], inputs.params[43]);
    break;
  case 45:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36], inputs.params[37],
        inputs.params[38], inputs.params[39], inputs.params[40],
        inputs.params[41], inputs.params[42], inputs.params[43],
        inputs.params[44]);
    break;
  case 46:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36], inputs.params[37],
        inputs.params[38], inputs.params[39], inputs.params[40],
        inputs.params[41], inputs.params[42], inputs.params[43],
        inputs.params[44], inputs.params[45]);
    break;
  case 47:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36], inputs.params[37],
        inputs.params[38], inputs.params[39], inputs.params[40],
        inputs.params[41], inputs.params[42], inputs.params[43],
        inputs.params[44], inputs.params[45], inputs.params[46]);
    break;
  case 48:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36], inputs.params[37],
        inputs.params[38], inputs.params[39], inputs.params[40],
        inputs.params[41], inputs.params[42], inputs.params[43],
        inputs.params[44], inputs.params[45], inputs.params[46],
        inputs.params[47]);
    break;
  case 49:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36], inputs.params[37],
        inputs.params[38], inputs.params[39], inputs.params[40],
        inputs.params[41], inputs.params[42], inputs.params[43],
        inputs.params[44], inputs.params[45], inputs.params[46],
        inputs.params[47], inputs.params[48]);
    break;
  case 50:
    wfn(output1, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36], inputs.params[37],
        inputs.params[38], inputs.params[39], inputs.params[40],
        inputs.params[41], inputs.params[42], inputs.params[43],
        inputs.params[44], inputs.params[45], inputs.params[46],
        inputs.params[47], inputs.params[48], inputs.params[49]);
    break;
  default:
    HPX_THROW_EXCEPTION(hpx::error::no_success,
                        "GenericComputeServer::execute_task",
                        "Error: number of task parameters not supported.");
  }
  outputs = {output1};
  break;
}
case 2: {
  void *output1;
  _dfr_checked_aligned_alloc(&output1, 512, inputs.output_sizes[0]);
  void *output2;
  _dfr_checked_aligned_alloc(&output2, 512, inputs.output_sizes[1]);
  switch (inputs.params.size()) {
  case 0:
    wfn(output1, output2);
    break;
  case 1:
    wfn(output1, output2, inputs.params[0]);
    break;
  case 2:
    wfn(output1, output2, inputs.params[0], inputs.params[1]);
    break;
  case 3:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2]);
    break;
  case 4:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3]);
    break;
  case 5:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4]);
    break;
  case 6:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5]);
    break;
  case 7:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6]);
    break;
  case 8:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7]);
    break;
  case 9:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8]);
    break;
  case 10:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9]);
    break;
  case 11:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10]);
    break;
  case 12:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11]);
    break;
  case 13:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12]);
    break;
  case 14:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13]);
    break;
  case 15:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14]);
    break;
  case 16:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15]);
    break;
  case 17:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16]);
    break;
  case 18:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17]);
    break;
  case 19:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18]);
    break;
  case 20:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19]);
    break;
  case 21:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20]);
    break;
  case 22:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21]);
    break;
  case 23:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22]);
    break;
  case 24:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23]);
    break;
  case 25:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24]);
    break;
  case 26:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25]);
    break;
  case 27:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26]);
    break;
  case 28:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27]);
    break;
  case 29:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28]);
    break;
  case 30:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29]);
    break;
  case 31:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30]);
    break;
  case 32:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31]);
    break;
  case 33:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32]);
    break;
  case 34:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33]);
    break;
  case 35:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34]);
    break;
  case 36:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35]);
    break;
  case 37:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36]);
    break;
  case 38:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36], inputs.params[37]);
    break;
  case 39:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36], inputs.params[37],
        inputs.params[38]);
    break;
  case 40:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36], inputs.params[37],
        inputs.params[38], inputs.params[39]);
    break;
  case 41:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36], inputs.params[37],
        inputs.params[38], inputs.params[39], inputs.params[40]);
    break;
  case 42:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36], inputs.params[37],
        inputs.params[38], inputs.params[39], inputs.params[40],
        inputs.params[41]);
    break;
  case 43:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36], inputs.params[37],
        inputs.params[38], inputs.params[39], inputs.params[40],
        inputs.params[41], inputs.params[42]);
    break;
  case 44:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36], inputs.params[37],
        inputs.params[38], inputs.params[39], inputs.params[40],
        inputs.params[41], inputs.params[42], inputs.params[43]);
    break;
  case 45:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36], inputs.params[37],
        inputs.params[38], inputs.params[39], inputs.params[40],
        inputs.params[41], inputs.params[42], inputs.params[43],
        inputs.params[44]);
    break;
  case 46:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36], inputs.params[37],
        inputs.params[38], inputs.params[39], inputs.params[40],
        inputs.params[41], inputs.params[42], inputs.params[43],
        inputs.params[44], inputs.params[45]);
    break;
  case 47:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36], inputs.params[37],
        inputs.params[38], inputs.params[39], inputs.params[40],
        inputs.params[41], inputs.params[42], inputs.params[43],
        inputs.params[44], inputs.params[45], inputs.params[46]);
    break;
  case 48:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36], inputs.params[37],
        inputs.params[38], inputs.params[39], inputs.params[40],
        inputs.params[41], inputs.params[42], inputs.params[43],
        inputs.params[44], inputs.params[45], inputs.params[46],
        inputs.params[47]);
    break;
  case 49:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36], inputs.params[37],
        inputs.params[38], inputs.params[39], inputs.params[40],
        inputs.params[41], inputs.params[42], inputs.params[43],
        inputs.params[44], inputs.params[45], inputs.params[46],
        inputs.params[47], inputs.params[48]);
    break;
  case 50:
    wfn(output1, output2, inputs.params[0], inputs.params[1], inputs.params[2],
        inputs.params[3], inputs.params[4], inputs.params[5], inputs.params[6],
        inputs.params[7], inputs.params[8], inputs.params[9], inputs.params[10],
        inputs.params[11], inputs.params[12], inputs.params[13],
        inputs.params[14], inputs.params[15], inputs.params[16],
        inputs.params[17], inputs.params[18], inputs.params[19],
        inputs.params[20], inputs.params[21], inputs.params[22],
        inputs.params[23], inputs.params[24], inputs.params[25],
        inputs.params[26], inputs.params[27], inputs.params[28],
        inputs.params[29], inputs.params[30], inputs.params[31],
        inputs.params[32], inputs.params[33], inputs.params[34],
        inputs.params[35], inputs.params[36], inputs.params[37],
        inputs.params[38], inputs.params[39], inputs.params[40],
        inputs.params[41], inputs.params[42], inputs.params[43],
        inputs.params[44], inputs.params[45], inputs.params[46],
        inputs.params[47], inputs.params[48], inputs.params[49]);
    break;
  default:
    HPX_THROW_EXCEPTION(hpx::error::no_success,
                        "GenericComputeServer::execute_task",
                        "Error: number of task parameters not supported.");
  }
  outputs = {output1, output2};
  break;
}
case 3: {
  void *output1;
  _dfr_checked_aligned_alloc(&output1, 512, inputs.output_sizes[0]);
  void *output2;
  _dfr_checked_aligned_alloc(&output2, 512, inputs.output_sizes[1]);
  void *output3;
  _dfr_checked_aligned_alloc(&output3, 512, inputs.output_sizes[2]);
  switch (inputs.params.size()) {
  case 0:
    wfn(output1, output2, output3);
    break;
  case 1:
    wfn(output1, output2, output3, inputs.params[0]);
    break;
  case 2:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1]);
    break;
  case 3:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2]);
    break;
  case 4:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3]);
    break;
  case 5:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4]);
    break;
  case 6:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5]);
    break;
  case 7:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6]);
    break;
  case 8:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7]);
    break;
  case 9:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8]);
    break;
  case 10:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9]);
    break;
  case 11:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10]);
    break;
  case 12:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11]);
    break;
  case 13:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12]);
    break;
  case 14:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13]);
    break;
  case 15:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14]);
    break;
  case 16:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15]);
    break;
  case 17:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16]);
    break;
  case 18:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17]);
    break;
  case 19:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18]);
    break;
  case 20:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19]);
    break;
  case 21:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20]);
    break;
  case 22:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21]);
    break;
  case 23:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22]);
    break;
  case 24:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23]);
    break;
  case 25:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23], inputs.params[24]);
    break;
  case 26:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23], inputs.params[24],
        inputs.params[25]);
    break;
  case 27:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23], inputs.params[24],
        inputs.params[25], inputs.params[26]);
    break;
  case 28:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23], inputs.params[24],
        inputs.params[25], inputs.params[26], inputs.params[27]);
    break;
  case 29:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23], inputs.params[24],
        inputs.params[25], inputs.params[26], inputs.params[27],
        inputs.params[28]);
    break;
  case 30:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23], inputs.params[24],
        inputs.params[25], inputs.params[26], inputs.params[27],
        inputs.params[28], inputs.params[29]);
    break;
  case 31:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23], inputs.params[24],
        inputs.params[25], inputs.params[26], inputs.params[27],
        inputs.params[28], inputs.params[29], inputs.params[30]);
    break;
  case 32:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23], inputs.params[24],
        inputs.params[25], inputs.params[26], inputs.params[27],
        inputs.params[28], inputs.params[29], inputs.params[30],
        inputs.params[31]);
    break;
  case 33:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23], inputs.params[24],
        inputs.params[25], inputs.params[26], inputs.params[27],
        inputs.params[28], inputs.params[29], inputs.params[30],
        inputs.params[31], inputs.params[32]);
    break;
  case 34:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23], inputs.params[24],
        inputs.params[25], inputs.params[26], inputs.params[27],
        inputs.params[28], inputs.params[29], inputs.params[30],
        inputs.params[31], inputs.params[32], inputs.params[33]);
    break;
  case 35:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23], inputs.params[24],
        inputs.params[25], inputs.params[26], inputs.params[27],
        inputs.params[28], inputs.params[29], inputs.params[30],
        inputs.params[31], inputs.params[32], inputs.params[33],
        inputs.params[34]);
    break;
  case 36:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23], inputs.params[24],
        inputs.params[25], inputs.params[26], inputs.params[27],
        inputs.params[28], inputs.params[29], inputs.params[30],
        inputs.params[31], inputs.params[32], inputs.params[33],
        inputs.params[34], inputs.params[35]);
    break;
  case 37:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23], inputs.params[24],
        inputs.params[25], inputs.params[26], inputs.params[27],
        inputs.params[28], inputs.params[29], inputs.params[30],
        inputs.params[31], inputs.params[32], inputs.params[33],
        inputs.params[34], inputs.params[35], inputs.params[36]);
    break;
  case 38:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23], inputs.params[24],
        inputs.params[25], inputs.params[26], inputs.params[27],
        inputs.params[28], inputs.params[29], inputs.params[30],
        inputs.params[31], inputs.params[32], inputs.params[33],
        inputs.params[34], inputs.params[35], inputs.params[36],
        inputs.params[37]);
    break;
  case 39:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23], inputs.params[24],
        inputs.params[25], inputs.params[26], inputs.params[27],
        inputs.params[28], inputs.params[29], inputs.params[30],
        inputs.params[31], inputs.params[32], inputs.params[33],
        inputs.params[34], inputs.params[35], inputs.params[36],
        inputs.params[37], inputs.params[38]);
    break;
  case 40:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23], inputs.params[24],
        inputs.params[25], inputs.params[26], inputs.params[27],
        inputs.params[28], inputs.params[29], inputs.params[30],
        inputs.params[31], inputs.params[32], inputs.params[33],
        inputs.params[34], inputs.params[35], inputs.params[36],
        inputs.params[37], inputs.params[38], inputs.params[39]);
    break;
  case 41:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23], inputs.params[24],
        inputs.params[25], inputs.params[26], inputs.params[27],
        inputs.params[28], inputs.params[29], inputs.params[30],
        inputs.params[31], inputs.params[32], inputs.params[33],
        inputs.params[34], inputs.params[35], inputs.params[36],
        inputs.params[37], inputs.params[38], inputs.params[39],
        inputs.params[40]);
    break;
  case 42:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23], inputs.params[24],
        inputs.params[25], inputs.params[26], inputs.params[27],
        inputs.params[28], inputs.params[29], inputs.params[30],
        inputs.params[31], inputs.params[32], inputs.params[33],
        inputs.params[34], inputs.params[35], inputs.params[36],
        inputs.params[37], inputs.params[38], inputs.params[39],
        inputs.params[40], inputs.params[41]);
    break;
  case 43:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23], inputs.params[24],
        inputs.params[25], inputs.params[26], inputs.params[27],
        inputs.params[28], inputs.params[29], inputs.params[30],
        inputs.params[31], inputs.params[32], inputs.params[33],
        inputs.params[34], inputs.params[35], inputs.params[36],
        inputs.params[37], inputs.params[38], inputs.params[39],
        inputs.params[40], inputs.params[41], inputs.params[42]);
    break;
  case 44:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23], inputs.params[24],
        inputs.params[25], inputs.params[26], inputs.params[27],
        inputs.params[28], inputs.params[29], inputs.params[30],
        inputs.params[31], inputs.params[32], inputs.params[33],
        inputs.params[34], inputs.params[35], inputs.params[36],
        inputs.params[37], inputs.params[38], inputs.params[39],
        inputs.params[40], inputs.params[41], inputs.params[42],
        inputs.params[43]);
    break;
  case 45:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23], inputs.params[24],
        inputs.params[25], inputs.params[26], inputs.params[27],
        inputs.params[28], inputs.params[29], inputs.params[30],
        inputs.params[31], inputs.params[32], inputs.params[33],
        inputs.params[34], inputs.params[35], inputs.params[36],
        inputs.params[37], inputs.params[38], inputs.params[39],
        inputs.params[40], inputs.params[41], inputs.params[42],
        inputs.params[43], inputs.params[44]);
    break;
  case 46:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23], inputs.params[24],
        inputs.params[25], inputs.params[26], inputs.params[27],
        inputs.params[28], inputs.params[29], inputs.params[30],
        inputs.params[31], inputs.params[32], inputs.params[33],
        inputs.params[34], inputs.params[35], inputs.params[36],
        inputs.params[37], inputs.params[38], inputs.params[39],
        inputs.params[40], inputs.params[41], inputs.params[42],
        inputs.params[43], inputs.params[44], inputs.params[45]);
    break;
  case 47:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23], inputs.params[24],
        inputs.params[25], inputs.params[26], inputs.params[27],
        inputs.params[28], inputs.params[29], inputs.params[30],
        inputs.params[31], inputs.params[32], inputs.params[33],
        inputs.params[34], inputs.params[35], inputs.params[36],
        inputs.params[37], inputs.params[38], inputs.params[39],
        inputs.params[40], inputs.params[41], inputs.params[42],
        inputs.params[43], inputs.params[44], inputs.params[45],
        inputs.params[46]);
    break;
  case 48:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23], inputs.params[24],
        inputs.params[25], inputs.params[26], inputs.params[27],
        inputs.params[28], inputs.params[29], inputs.params[30],
        inputs.params[31], inputs.params[32], inputs.params[33],
        inputs.params[34], inputs.params[35], inputs.params[36],
        inputs.params[37], inputs.params[38], inputs.params[39],
        inputs.params[40], inputs.params[41], inputs.params[42],
        inputs.params[43], inputs.params[44], inputs.params[45],
        inputs.params[46], inputs.params[47]);
    break;
  case 49:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23], inputs.params[24],
        inputs.params[25], inputs.params[26], inputs.params[27],
        inputs.params[28], inputs.params[29], inputs.params[30],
        inputs.params[31], inputs.params[32], inputs.params[33],
        inputs.params[34], inputs.params[35], inputs.params[36],
        inputs.params[37], inputs.params[38], inputs.params[39],
        inputs.params[40], inputs.params[41], inputs.params[42],
        inputs.params[43], inputs.params[44], inputs.params[45],
        inputs.params[46], inputs.params[47], inputs.params[48]);
    break;
  case 50:
    wfn(output1, output2, output3, inputs.params[0], inputs.params[1],
        inputs.params[2], inputs.params[3], inputs.params[4], inputs.params[5],
        inputs.params[6], inputs.params[7], inputs.params[8], inputs.params[9],
        inputs.params[10], inputs.params[11], inputs.params[12],
        inputs.params[13], inputs.params[14], inputs.params[15],
        inputs.params[16], inputs.params[17], inputs.params[18],
        inputs.params[19], inputs.params[20], inputs.params[21],
        inputs.params[22], inputs.params[23], inputs.params[24],
        inputs.params[25], inputs.params[26], inputs.params[27],
        inputs.params[28], inputs.params[29], inputs.params[30],
        inputs.params[31], inputs.params[32], inputs.params[33],
        inputs.params[34], inputs.params[35], inputs.params[36],
        inputs.params[37], inputs.params[38], inputs.params[39],
        inputs.params[40], inputs.params[41], inputs.params[42],
        inputs.params[43], inputs.params[44], inputs.params[45],
        inputs.params[46], inputs.params[47], inputs.params[48],
        inputs.params[49]);
    break;
  default:
    HPX_THROW_EXCEPTION(hpx::error::no_success,
                        "GenericComputeServer::execute_task",
                        "Error: number of task parameters not supported.");
  }
  outputs = {output1, output2, output3};
  break;
}
