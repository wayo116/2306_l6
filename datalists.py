import numpy as np

n_dlists=1500+50
dlists=[\
[10,22,25,30,31,39],\
[12,15,28,31,34,43],\
[4,5,10,19,26,37],\
[4,6,9,24,26,29],\
[8,11,15,31,33,38],\
[4,9,15,21,26,33],\
[2,5,26,32,34,40],\
[3,13,23,28,30,43],\
[16,17,27,34,38,42],\
[13,17,18,22,23,41],\
[11,20,23,24,31,33],\
[14,21,30,38,40,42],\
[7,13,34,37,42,43],\
[2,7,20,25,35,40],\
[1,14,17,27,32,43],\
[4,10,23,29,31,36],\
[12,15,16,17,28,42],\
[1,6,9,16,18,22],\
[1,13,18,23,38,41],\
[7,14,18,19,22,25],\
[1,10,13,17,20,41],\
[6,7,13,18,25,43],\
[4,10,18,25,34,38],\
[5,18,21,22,26,42],\
[9,11,12,14,27,38],\
[5,10,30,35,36,41],\
[5,8,18,20,24,34],\
[2,11,23,25,34,36],\
[6,24,30,33,37,38],\
[2,21,25,34,35,40],\
[3,8,10,25,26,29],\
[1,2,13,21,23,31],\
[5,22,26,34,39,42],\
[10,13,18,28,36,39],\
[2,4,6,13,20,38],\
[4,17,26,30,31,43],\
[2,5,10,22,24,42],\
[2,10,27,38,40,41],\
[1,2,8,9,10,20],\
[11,30,35,38,40,43],\
[4,22,27,31,32,40],\
[12,14,18,25,36,42],\
[9,16,18,19,39,41],\
[1,9,34,35,39,42],\
[18,25,26,37,39,43],\
[6,12,13,15,17,41],\
[11,18,34,40,41,42],\
[4,5,19,32,34,37],\
[15,23,26,34,41,43],\
[2,5,15,16,23,26],\
[26,30,31,35,41,43],\
[24,29,32,35,36,40],\
[1,6,14,15,23,24],\
[1,4,16,31,32,41],\
[8,9,18,19,30,36],\
[6,14,18,22,29,30],\
[6,9,16,26,29,34],\
[3,13,16,19,26,39],\
[2,13,17,19,41,43],\
[2,5,6,26,33,35],\
[4,11,24,35,37,42],\
[3,19,30,32,36,40],\
[5,14,19,23,30,42],\
[7,31,32,36,39,42],\
[10,18,32,39,41,42],\
[1,5,19,26,35,37],\
[11,13,20,33,35,40],\
[9,19,24,25,26,39],\
[5,16,19,21,30,35],\
[5,11,18,25,33,38],\
[4,5,9,26,27,29],\
[17,18,26,30,36,40],\
[12,21,22,27,31,32],\
[5,21,22,30,32,36],\
[3,6,12,16,31,42],\
[1,11,18,24,37,43],\
[4,11,15,19,28,32],\
[6,21,30,32,39,40],\
[7,15,17,21,32,35],\
[16,19,25,26,35,43],\
[1,10,15,19,20,32],\
[16,21,26,31,39,42],\
[2,6,20,30,31,41],\
[1,6,7,20,37,40],\
[5,7,13,21,23,30],\
[8,14,18,38,39,42],\
[1,10,25,27,28,33],\
[7,14,28,32,42,43],\
[6,12,15,35,36,39],\
[19,25,29,36,37,41],\
[4,11,13,26,27,36],\
[3,6,7,19,35,39],\
[1,2,4,9,22,25],\
[2,5,15,20,22,36],\
[1,12,21,24,29,42],\
[3,15,18,21,34,40],\
[1,4,13,17,31,34],\
[8,11,23,28,36,39],\
[1,9,11,14,15,40],\
[3,11,14,22,27,32],\
[7,9,10,21,39,40],\
[1,8,21,25,26,35],\
[3,17,18,19,20,42],\
[16,20,22,29,33,40],\
[4,14,17,19,22,42],\
[14,21,24,27,31,38],\
[1,7,12,36,37,39],\
[7,12,20,31,33,43],\
[12,14,22,24,29,33],\
[6,9,25,26,27,32],\
[1,21,24,33,40,41],\
[5,6,11,19,28,35],\
[22,24,25,27,29,39],\
[3,18,19,34,38,42],\
[3,7,19,21,32,35],\
[4,8,16,21,22,26],\
[11,15,24,25,28,31],\
[4,8,14,18,26,38],\
[3,13,16,22,33,42],\
[9,14,16,21,27,43],\
[2,10,16,17,20,28],\
[6,9,24,26,33,37],\
[6,10,15,16,30,38],\
[2,5,30,33,36,42],\
[7,8,14,17,22,39],\
[1,5,18,25,32,38],\
[4,11,15,16,35,39],\
[1,6,9,15,18,38],\
[15,21,23,24,33,36],\
[2,26,27,32,38,42],\
[2,6,15,16,32,37],\
[11,15,20,26,33,42],\
[6,32,38,39,40,42],\
[1,5,15,24,25,28],\
[3,32,33,34,35,40],\
[1,2,21,27,32,37],\
[3,4,7,20,22,41],\
[1,20,29,35,40,41],\
[14,16,28,34,35,36],\
[13,15,32,35,37,38],\
[3,6,9,28,37,41],\
[1,5,18,26,39,40],\
[14,22,23,30,33,38],\
[2,6,7,18,27,38],\
[5,8,27,30,33,39],\
[11,19,20,27,31,35],\
[4,6,16,27,39,42],\
[6,16,18,33,42,43],\
[2,3,11,12,22,35],\
[2,4,13,18,32,35],\
[3,8,14,18,28,37],\
[2,12,14,26,32,37],\
[7,8,21,23,34,36],\
[4,8,17,31,37,42],\
[13,16,18,29,33,41],\
[8,16,21,23,29,37],\
[4,7,21,22,34,42],\
[2,4,10,24,34,42],\
[20,21,23,26,29,35],\
[7,8,9,23,25,41],\
[1,26,27,28,33,37],\
[5,10,26,30,34,40],\
[6,15,28,30,32,43],\
[6,9,10,21,33,41],\
[1,7,11,13,15,42],\
[4,19,29,32,35,42],\
[10,12,13,21,31,41],\
[3,15,17,21,22,23],\
[11,20,26,31,34,38],\
[5,13,21,27,31,36],\
[2,3,17,27,28,34],\
[14,19,25,29,33,43],\
[24,33,35,36,41,43],\
[6,17,22,27,37,42],\
[16,27,37,38,39,43],\
[1,16,20,23,32,37],\
[2,4,13,14,22,29],\
[5,16,21,24,31,34],\
[5,8,18,25,40,43],\
[14,16,26,37,41,43],\
[19,22,24,29,34,41],\
[7,10,17,20,21,27],\
[1,6,19,21,27,39],\
[4,17,18,25,26,27],\
[6,14,31,35,38,43],\
[3,19,29,35,42,43],\
[4,10,23,25,37,42],\
[10,11,19,29,30,36],\
[11,13,16,19,20,38],\
[11,12,14,28,36,38],\
[21,23,31,38,42,43],\
[9,10,14,21,22,33],\
[2,7,10,12,29,36],\
[8,15,16,23,38,43],\
[7,12,15,21,27,32],\
[1,3,15,33,39,43],\
[8,16,19,23,29,41],\
[2,3,22,26,35,42],\
[3,8,14,15,19,35],\
[3,19,21,29,39,43],\
[2,8,13,14,33,40],\
[1,9,14,28,32,40],\
[10,11,12,15,16,40],\
[5,15,16,24,34,39],\
[2,5,7,12,16,20],\
[8,9,19,21,27,30],\
[5,9,13,20,26,32],\
[21,25,26,35,39,40],\
[2,3,11,16,27,36],\
[1,12,24,26,30,36],\
[4,8,19,32,36,43],\
[18,19,20,23,29,35],\
[3,7,9,26,35,36],\
[1,9,24,28,31,41],\
[7,15,23,26,30,35],\
[10,11,24,33,34,36],\
[19,23,30,31,32,40],\
[1,3,10,12,26,42],\
[21,22,23,26,26,38],\
[9,11,19,23,30,43],\
[1,4,8,11,30,42],\
[3,5,17,25,33,39],\
[7,12,15,17,19,29],\
[16,29,34,35,40,41],\
[3,10,22,23,26,39],\
[7,15,19,24,25,28],\
[3,5,17,20,24,38],\
[2,19,21,24,26,41],\
[1,4,6,21,34,41],\
[2,5,6,13,19,29],\
[4,5,20,29,40,43],\
[9,17,22,25,30,41],\
[5,8,12,16,27,38],\
[3,14,16,24,32,36],\
[1,8,13,22,41,42],\
[4,12,16,34,35,41],\
[10,18,22,23,25,35],\
[10,13,14,15,17,22],\
[18,23,25,26,32,34],\
[2,5,8,20,29,40],\
[12,21,23,36,37,38],\
[4,6,32,35,40,42],\
[1,2,7,14,32,42],\
[6,13,17,18,21,41],\
[7,14,22,35,37,39],\
[1,2,20,22,35,38],\
[9,11,26,28,39,43],\
[2,3,8,17,23,32],\
[10,15,30,35,42,43],\
[16,21,30,32,40,43],\
[1,5,6,13,34,36],\
[6,17,18,35,37,38],\
[4,16,25,30,36,43],\
[7,9,18,21,26,43],\
[10,12,15,29,33,42],\
[11,21,25,31,36,39],\
[6,16,27,31,32,34],\
[16,21,22,27,36,43],\
[2,13,19,27,40,42],\
[2,6,10,13,16,32],\
[13,15,18,30,37,43],\
[5,7,16,34,39,41],\
[18,32,35,36,39,41],\
[6,12,30,31,41,43],\
[5,7,11,12,24,31],\
[23,29,32,33,37,42],\
[3,15,17,19,21,34],\
[13,14,15,28,37,39],\
[5,7,10,17,29,32],\
[2,8,15,24,32,39],\
[1,2,3,20,31,32],\
[2,10,16,38,39,43],\
[4,6,10,19,28,38],\
[1,2,5,25,40,41],\
[1,19,22,23,31,36],\
[2,8,29,31,32,33],\
[2,5,7,10,16,34],\
[5,7,19,26,37,40],\
[12,13,20,24,30,35],\
[5,9,17,28,30,37],\
[5,6,18,20,27,37],\
[1,12,20,26,35,37],\
[2,4,15,16,26,33],\
[12,15,26,34,35,38],\
[7,19,24,32,40,42],\
[4,15,17,20,26,31],\
[1,12,25,27,39,42],\
[5,21,22,23,26,39],\
[10,19,28,30,38,42],\
[2,8,17,23,24,27],\
[6,14,16,27,40,42],\
[7,8,16,32,41,42],\
[5,9,15,30,32,39],\
[3,14,18,24,38,43],\
[8,11,22,32,37,42],\
[4,5,27,28,35,41],\
[18,24,26,35,39,42],\
[4,11,14,17,22,39],\
[1,18,19,24,31,40],\
[8,14,15,26,37,43],\
[3,11,25,26,39,42],\
[17,21,26,33,36,38],\
[11,12,16,17,20,36],\
[12,26,28,32,33,35],\
[1,20,25,28,33,37],\
[13,29,33,37,41,42],\
[8,14,19,33,35,40],\
[6,13,14,32,33,37],\
[2,5,9,30,37,40],\
[1,9,14,30,35,38],\
[1,9,18,22,24,43],\
[15,22,27,37,38,41],\
[2,9,14,16,17,21],\
[2,6,7,10,30,33],\
[2,5,10,34,37,40],\
[3,19,26,28,33,40],\
[1,14,17,34,35,37],\
[2,5,9,11,27,32],\
[4,7,19,21,39,41],\
[3,7,20,21,28,32],\
[2,3,15,24,30,43],\
[2,15,17,25,34,38],\
[5,17,21,22,32,35],\
[17,21,22,40,41,42],\
[1,6,11,13,30,33],\
[5,8,20,21,27,43],\
[6,15,16,22,36,41],\
[14,25,27,30,39,41],\
[25,32,34,37,41,42],\
[5,9,17,21,27,29],\
[11,16,22,29,37,43],\
[13,14,18,19,25,42],\
[1,11,12,24,32,38],\
[11,16,22,24,38,40],\
[8,13,17,18,21,38],\
[6,11,12,34,36,41],\
[9,13,28,29,32,40],\
[19,20,22,30,35,43],\
[10,12,19,28,35,41],\
[1,2,14,15,21,34],\
[5,7,9,21,42,43],\
[1,8,16,35,40,42],\
[1,17,25,28,32,42],\
[6,10,13,25,31,38],\
[19,21,22,28,29,30],\
[10,11,27,28,29,37],\
[1,6,7,9,31,37],\
[6,22,28,30,35,37],\
[5,9,13,16,17,34],\
[18,24,32,34,37,41],\
[8,15,17,24,26,38],\
[11,18,24,27,37,38],\
[4,16,18,32,38,42],\
[8,11,23,31,34,37],\
[7,19,24,29,39,43],\
[4,6,11,13,23,32],\
[4,10,11,17,34,37],
[5,7,9,17,24,35],\
[18,24,29,33,39,42],\
[9,10,18,27,28,35],\
[4,12,17,18,23,41],\
[3,10,20,23,29,43],\
[8,26,28,32,41,43],\
[3,9,21,31,41,42],\
[2,4,5,6,24,34],\
[1,20,30,32,35,37],\
[7,12,15,21,40,42],\
[4,13,14,25,27,32],\
[6,22,38,40,41,42],\
[6,7,9,20,21,27],\
[4,6,8,22,27,32],\
[1,5,6,27,33,38],\
[1,6,7,9,23,40],\
[1,3,8,32,37,43],\
[4,8,14,16,39,40],\
[4,9,10,13,24,27],\
[4,8,15,24,25,31],\
[3,14,16,25,26,38],\
[6,10,15,18,32,40],\
[11,15,16,24,28,38],\
[8,13,14,18,29,31],\
[18,29,32,34,35,39],\
[4,18,19,28,33,36],\
[5,7,8,18,22,26],\
[3,6,10,15,29,30],\
[2,17,19,24,28,29],\
[4,16,19,22,26,29],\
[1,3,11,17,20,37],\
[2,18,19,24,29,31],\
[1,10,12,22,34,43],\
[3,15,19,34,37,40],\
[10,22,26,27,28,41],\
[17,19,29,31,37,39],\
[2,6,10,20,34,43],\
[24,25,26,27,34,36],\
[6,19,20,26,34,38],\
[10,12,25,29,32,41],\
[2,9,12,13,14,22],\
[2,5,11,20,21,32],\
[3,14,24,29,30,42],\
[11,20,24,25,30,37],\
[2,4,8,10,22,33],\
[12,21,35,36,40,43],\
[12,22,24,33,35,42],\
[1,7,20,31,41,42],\
[11,17,23,28,34,41],\
[2,5,6,20,33,38],\
[3,11,24,25,37,40],\
[3,7,17,18,30,36],\
[2,4,16,21,22,40],\
[2,20,30,38,39,40],\
[3,5,12,16,38,42],\
[4,5,15,32,33,43],\
[2,27,28,33,38,40],\
[4,7,9,12,20,28],\
[4,8,13,18,21,23],\
[16,18,19,34,35,36],\
[1,7,25,26,38,41],\
[4,13,17,19,30,43],\
[1,24,28,33,35,42],\
[6,7,25,26,33,36],\
[11,24,29,36,37,42],\
[12,14,17,20,24,30],\
[2,14,20,23,37,43],\
[1,2,14,16,22,28],\
[5,10,11,12,21,37],\
[4,21,32,35,42,43],\
[3,10,25,26,27,31],\
[2,9,10,15,24,36],\
[1,3,7,34,39,40],\
[12,19,21,26,40,41],\
[7,13,17,21,24,26],\
[2,19,24,25,32,42],\
[2,5,7,20,25,28],\
[8,14,17,25,36,41],\
[5,6,8,20,22,25],\
[2,17,24,26,34,43],\
[10,13,19,28,33,39],\
[20,24,27,29,34,35],\
[3,4,6,19,23,24],\
[2,17,29,37,38,39],\
[1,3,10,27,34,41],\
[10,24,26,33,36,42],\
[2,13,23,24,25,33],\
[8,10,26,31,34,39],\
[7,20,21,33,41,43],\
[23,29,34,35,40,43],\
[9,18,19,21,25,26],\
[3,7,11,15,23,41],\
[5,16,18,27,30,43],\
[8,9,35,36,42,43],\
[7,11,26,32,42,43],\
[6,9,12,19,40,42],\
[3,12,23,31,36,42],\
[10,11,13,28,34,39],\
[6,16,21,23,26,42],\
[9,10,12,15,29,37],\
[11,27,29,30,37,38],\
[4,11,14,15,20,38],\
[9,20,28,35,41,43],\
[3,6,13,32,38,43],\
[2,4,9,15,18,39],\
[8,9,12,25,26,28],\
[2,4,5,16,19,38],\
[1,3,24,26,29,33],\
[1,2,19,21,24,37],\
[1,10,15,24,35,43],\
[1,7,13,24,40,43],\
[2,5,22,31,32,43],\
[2,5,21,22,31,38],\
[1,4,5,19,23,35],\
[1,10,13,18,20,41],\
[3,5,10,21,33,39],\
[4,5,6,7,24,28],\
[8,22,24,30,40,41],\
[12,15,25,32,41,43],\
[6,11,17,20,34,35],\
[3,8,11,14,22,41],\
[14,15,17,29,31,39],\
[4,7,14,20,31,41],\
[5,13,14,16,25,27],\
[2,5,23,36,37,41],\
[8,16,19,28,36,39],\
[3,4,12,18,28,38],\
[4,9,10,16,17,42],\
[6,8,14,18,25,41],\
[9,15,23,27,29,40],\
[2,3,8,15,19,28],\
[4,6,10,16,24,43],\
[5,15,27,29,34,37],\
[4,25,27,28,32,41],\
[2,11,12,19,21,24],\
[5,6,9,15,19,36],\
[8,14,18,22,28,29],\
[7,8,9,10,20,38],\
[2,16,19,32,34,42],\
[1,2,14,19,21,33],\
[8,10,11,18,36,43],\
[5,10,22,27,38,42],\
[1,15,26,27,29,33],\
[8,16,20,28,37,43],\
[2,8,10,12,19,23],\
[1,4,9,16,23,31],\
[9,12,15,16,26,39],\
[6,15,18,20,26,29],\
[1,10,15,33,34,42],\
[2,6,23,25,30,31],\
[3,6,21,28,29,34],\
[4,5,11,19,31,34],\
[10,16,19,34,35,39],\
[1,2,7,14,29,43],\
[2,9,15,22,26,35],\
[5,15,18,22,33,40],\
[2,14,30,31,38,39],\
[2,19,24,32,36,43],\
[11,15,18,22,37,42],\
[7,9,23,32,37,40],\
[2,3,13,22,23,30],\
[2,3,9,12,30,41],\
[8,18,23,26,33,38],\
[1,7,8,17,25,42],\
[5,6,10,24,29,40],\
[2,3,13,28,32,37],\
[10,24,31,40,41,43],\
[4,6,12,16,17,34],\
[9,10,19,21,28,30],\
[2,3,7,14,19,41],\
[6,10,13,19,32,43],\
[6,16,26,35,36,39],\
[3,8,23,30,32,40],\
[3,10,20,21,27,33],\
[5,12,21,31,36,43],\
[13,24,25,26,32,39],\
[12,23,26,29,36,37],\
[9,24,25,27,34,36],\
[24,32,34,36,41,43],\
[6,9,10,15,34,39],\
[6,25,27,28,30,43],\
[7,11,23,36,38,41],\
[4,6,19,36,38,40],\
[6,25,30,38,40,42],\
[7,10,24,33,35,42],\
[4,7,25,30,34,43],\
[6,7,9,16,30,41],\
[3,18,25,26,32,33],\
[3,12,24,28,38,39],\
[4,6,15,22,30,42],\
[1,2,7,29,34,42],\
[5,11,15,16,23,32],\
[14,23,31,37,39,43],\
[7,14,18,26,28,39],\
[2,5,15,27,30,35],\
[1,18,29,30,31,37],\
[2,6,8,12,13,17],\
[18,24,27,32,35,41],\
[6,9,16,24,29,37],\
[8,9,19,24,35,43],\
[9,21,23,34,37,38],\
[11,32,37,39,41,42],\
[7,8,13,36,38,42],\
[4,14,17,21,24,39],\
[1,26,27,30,37,38],\
[1,4,10,11,24,28],\
[3,7,9,15,29,40],\
[1,9,11,21,29,36],\
[4,10,16,17,20,24],\
[3,4,18,28,34,36],\
[12,16,20,22,31,43],\
[5,6,9,11,16,35],\
[7,19,21,28,40,41],\
[3,15,18,23,31,37],\
[18,20,22,26,34,37],\
[4,9,13,32,37,38],\
[1,11,14,27,39,42],\
[9,10,24,30,35,40],\
[9,10,11,33,39,42],\
[8,11,15,22,35,36],\
[4,18,31,33,34,41],\
[3,12,16,23,25,28],\
[5,20,26,27,28,33],\
[1,6,27,33,35,43],\
[3,8,13,14,31,37],\
[4,11,15,20,24,43],\
[9,20,26,29,30,34],\
[3,8,16,19,25,26],\
[10,16,20,26,33,36],\
[8,14,20,25,31,41],\
[1,11,19,23,32,37],\
[2,5,16,32,35,41],\
[2,14,15,27,40,43],\
[7,8,11,15,36,39],\
[1,12,14,24,33,37],\
[4,15,18,19,22,29],\
[21,28,31,33,34,40],\
[3,10,18,22,23,40],\
[5,8,15,20,25,27],\
[2,6,14,28,34,37],\
[4,7,17,18,26,36],\
[4,10,11,23,36,37],\
[13,15,27,30,31,39],\
[10,15,17,18,20,41],\
[23,24,32,37,41,42],\
[12,22,28,34,37,42],\
[4,7,15,17,23,25],\
[15,18,22,36,37,39],\
[2,19,27,28,30,38],\
[8,14,28,29,33,40],\
[1,4,6,35,38,40],\
[1,3,11,26,38,39],\
[8,17,24,34,36,42],\
[7,11,16,20,30,40],\
[2,7,12,21,29,33],\
[5,10,14,31,35,38],\
[1,14,16,24,32,36],\
[1,2,6,9,17,38],\
[10,28,35,37,39,43],\
[6,18,19,25,28,40],\
[1,8,14,28,40,43],\
[7,8,28,34,37,41],\
[19,22,25,30,33,34],\
[5,18,33,38,41,43],\
[12,22,28,30,36,43],\
[1,5,7,11,29,39],\
[3,10,11,31,33,36],\
[9,15,21,23,32,42],\
[14,22,24,36,39,42],\
[5,8,13,20,28,42],\
[14,16,17,34,36,41],\
[8,14,25,39,42,43],\
[3,6,9,15,21,42],\
[2,19,23,30,33,36],\
[3,13,22,26,30,39],\
[2,16,27,35,40,42],\
[3,8,24,25,26,28],\
[5,6,12,16,23,36],\
[11,12,24,32,38,40],\
[5,14,20,30,38,41],\
[6,16,17,20,22,43],\
[11,19,20,26,32,33],\
[12,25,26,32,35,42],\
[10,20,21,24,39,43],\
[2,4,6,37,39,42],\
[9,12,15,19,26,39],\
[1,15,19,22,26,28],\
[2,8,10,12,31,42],\
[9,10,11,20,28,40],\
[1,14,20,26,27,34],\
[3,24,28,36,37,39],\
[6,7,19,31,39,41],\
[1,4,17,22,35,42],\
[12,18,20,23,33,40],\
[1,2,7,8,10,42],\
[15,17,18,22,30,39],\
[3,5,11,20,25,32],\
[1,3,13,29,38,39],\
[8,24,28,30,38,41],\
[3,9,19,20,23,30],\
[3,13,18,20,24,31],\
[8,10,20,25,38,43],\
[6,15,18,26,27,31],\
[2,6,21,30,31,36],\
[2,19,22,25,32,36],\
[5,6,10,13,24,33],\
[15,16,17,29,31,32],\
[2,4,5,13,18,21],\
[5,12,27,36,39,40],\
[16,17,20,22,29,40],\
[2,5,21,22,23,38],\
[2,9,18,23,29,32],\
[2,4,6,21,27,37],\
[7,8,9,25,29,31],\
[5,7,11,16,33,38],\
[3,10,31,32,38,39],\
[9,10,13,21,34,36],\
[1,8,13,19,23,28],\
[19,24,25,36,37,43],\
[1,3,8,14,35,39],\
[6,13,20,25,36,42],\
[4,8,17,20,29,41],\
[18,20,21,24,40,42],\
[4,7,11,33,34,37],\
[11,17,19,24,33,41],\
[3,4,8,12,24,37],\
[1,2,12,16,19,40],\
[9,22,31,32,34,43],\
[1,4,15,16,23,30],\
[22,24,27,30,32,40],\
[5,17,20,28,39,42],\
[3,6,19,31,35,40],\
[14,19,22,23,28,43],\
[8,12,30,36,41,42],\
[4,15,20,24,26,30],\
[6,8,10,33,35,43],\
[5,23,25,26,29,31],\
[7,11,17,27,32,40],\
[2,14,22,23,35,43],\
[3,5,16,23,31,43],\
[2,3,10,16,26,27],\
[1,10,19,23,26,34],\
[1,4,15,18,25,28],\
[7,13,14,33,39,43],\
[19,21,31,39,42,43],\
[12,13,19,23,28,35],\
[1,4,10,14,24,26],\
[11,21,30,31,36,37],\
[4,6,14,17,23,33],\
[8,15,21,24,35,38],\
[1,18,21,31,37,41],\
[3,4,7,8,22,33],\
[20,22,28,29,32,41],\
[5,6,12,14,30,38],\
[3,18,19,20,37,38],\
[4,5,24,29,33,35],\
[6,12,25,36,38,41],\
[1,11,37,38,39,42],\
[6,11,19,28,30,43],\
[3,15,21,29,30,35],\
[9,15,17,21,23,42],\
[4,5,26,30,34,42],\
[2,16,19,20,24,36],\
[1,16,24,28,40,42],\
[4,8,23,29,34,35],\
[22,24,29,31,32,34],\
[2,8,31,38,40,42],\
[10,12,13,29,35,39],\
[3,6,15,21,42,43],\
[1,11,21,25,34,36],\
[2,6,7,9,19,23],\
[1,11,33,34,35,38],\
[3,10,12,26,38,42],\
[2,7,9,28,38,39],\
[1,12,14,26,34,41],\
[6,9,15,19,39,42],\
[6,16,17,27,33,41],\
[1,6,21,34,36,39],\
[6,8,14,27,36,42],\
[12,25,29,32,38,43],\
[6,7,12,22,30,33],\
[1,4,8,10,18,33],\
[12,15,25,31,39,40],\
[15,25,32,33,36,38],\
[8,13,14,18,24,38],\
[3,6,9,13,36,37],\
[4,5,7,10,13,16],\
[2,6,16,23,32,35],\
[6,8,12,17,18,24],\
[1,19,31,33,35,43],\
[1,5,8,14,19,21],\
[6,19,21,31,38,41],\
[3,6,15,18,23,27],\
[6,7,20,24,28,43],\
[4,19,26,40,42,43],\
[2,11,18,27,30,33],\
[21,22,24,26,28,30],\
[3,10,15,22,27,40],\
[5,17,19,22,41,42],\
[5,8,16,24,32,33],\
[10,13,26,27,30,36],\
[3,7,9,31,38,43],\
[3,20,21,30,34,35],\
[1,7,20,28,34,38],\
[10,13,15,18,25,38],\
[9,12,19,21,39,40],\
[6,8,13,17,18,20],\
[6,15,16,19,38,41],\
[8,12,19,24,30,36],\
[3,12,20,33,35,41],\
[1,4,9,23,30,36],\
[21,22,24,30,31,43],\
[7,8,21,24,34,42],\
[8,9,16,27,30,33],\
[1,3,5,8,20,39],\
[3,17,20,32,36,43],\
[1,10,25,27,31,40],\
[11,21,25,34,35,36],\
[12,17,26,30,33,40],\
[3,9,21,34,35,42],\
[5,9,25,27,28,33],\
[2,11,29,36,40,42],\
[11,13,27,37,41,42],\
[4,13,18,27,30,39],\
[6,12,13,34,38,42],\
[2,7,28,39,42,43],\
[1,12,14,16,20,29],\
[5,10,16,17,32,37],\
[8,26,27,29,35,43],\
[11,12,17,23,29,43],\
[2,11,19,21,28,38],\
[14,16,19,29,32,37],\
[5,11,20,25,26,32],\
[7,9,10,19,28,33],\
[4,11,23,27,35,42],\
[6,26,34,37,42,43],\
[5,13,20,28,30,32],\
[3,8,14,28,31,43],\
[3,6,15,23,24,31],\
[26,27,30,31,34,40],\
[5,9,13,31,32,35],\
[6,13,17,18,27,43],\
[1,19,21,30,31,43],\
[3,7,16,26,34,39],\
[21,29,30,32,38,42],\
[4,10,11,12,18,25],\
[14,22,27,29,33,37],\
[7,13,15,21,23,34],\
[6,9,20,34,37,41],\
[3,7,24,25,27,39],\
[1,3,12,13,21,32],\
[2,8,17,27,30,31],\
[9,15,17,19,27,33],\
[14,20,24,25,31,32],\
[1,15,21,24,37,41],\
[11,20,21,30,36,43],\
[2,4,7,22,32,35],\
[4,9,17,20,23,42],\
[5,13,22,31,37,39],\
[7,10,18,21,28,36],\
[13,16,17,19,20,38],\
[1,8,11,17,22,26],\
[10,17,20,21,36,43],\
[2,4,5,20,37,43],\
[9,17,24,25,33,40],\
[3,15,19,30,34,37],\
[3,30,32,33,34,39],\
[2,25,27,28,35,42],\
[8,14,15,22,25,40],\
[3,22,25,28,32,39],\
[1,4,24,30,39,40],\
[1,14,31,32,39,42],\
[9,14,20,23,27,32],\
[2,3,9,19,40,42],\
[3,20,36,38,39,41],\
[9,11,19,22,27,38],\
[23,26,27,29,30,43],\
[6,12,16,24,25,37],\
[3,19,34,36,40,43],\
[11,14,16,21,24,39],\
[3,8,17,25,28,43],\
[7,8,16,31,32,33],\
[5,13,20,24,25,27],\
[3,21,23,24,27,32],\
[13,22,26,34,38,43],\
[8,14,22,25,27,41],\
[9,12,15,16,18,35],\
[1,15,21,22,26,29],\
[8,15,17,30,32,36],\
[1,3,12,19,26,29],\
[3,5,8,11,33,43],\
[1,3,14,26,27,42],\
[7,11,17,21,37,39],\
[18,22,29,31,39,43],\
[1,5,10,32,38,40],\
[4,6,8,22,27,38],\
[3,18,21,25,29,37],\
[4,12,16,27,28,41],\
[5,7,31,33,36,39],\
[5,6,10,22,34,41],\
[3,14,15,24,25,36],\
[4,11,17,30,36,43],\
[3,11,19,26,29,38],\
[14,23,24,28,37,38],\
[15,19,32,35,36,38],\
[15,19,21,22,29,40],\
[14,19,22,23,31,37],\
[15,20,23,35,38,42],\
[9,12,16,18,27,38],\
[3,11,19,30,38,42],\
[10,15,17,27,42,43],\
[6,13,22,28,42,43],\
[3,11,13,27,29,36],\
[2,3,29,30,32,43],\
[1,5,6,12,30,31],\
[2,7,17,18,21,40],\
[2,3,4,12,22,29],\
[2,4,18,23,31,38],\
[3,9,19,22,23,40],\
[6,10,15,20,23,26],\
[1,11,26,30,32,40],\
[10,22,31,34,36,43],\
[10,12,19,20,21,22],\
[2,21,23,29,30,43],\
[4,23,29,31,32,35],\
[4,6,10,24,32,37],\
[5,11,17,18,19,25],\
[5,18,23,32,35,39],\
[1,16,18,28,36,38],\
[9,18,23,24,26,30],\
[2,11,22,32,36,42],\
[1,4,6,22,24,38],\
[14,19,27,28,37,42],\
[3,23,24,28,29,42],\
[2,9,17,38,39,42],\
[10,16,22,25,31,39],\
[11,15,22,23,26,30],\
[2,6,16,24,30,41],\
[5,19,26,36,39,43],\
[7,16,17,28,39,43],\
[1,6,21,23,27,40],\
[2,9,12,16,31,36],\
[3,10,15,23,36,37],\
[7,20,25,29,33,37],\
[9,10,14,15,39,43],\
[2,3,14,27,28,38],\
[5,12,22,29,38,39],\
[1,18,28,31,35,37],\
[8,11,15,17,26,43],\
[5,9,12,19,30,35],\
[13,15,21,25,36,41],\
[1,3,12,16,20,35],\
[2,12,14,26,38,43],\
[3,7,16,17,20,32],\
[5,9,20,21,31,41],\
[4,15,30,37,42,43],\
[14,18,19,24,25,33],\
[10,12,37,39,40,43],\
[10,12,15,25,31,34],\
[1,21,23,26,31,33],\
[2,29,30,31,33,34],\
[16,26,32,35,37,39],\
[25,26,27,28,32,37],\
[13,20,24,32,35,42],\
[5,13,30,31,32,35],\
[4,6,19,32,33,38],\
[5,7,26,36,38,39],\
[9,17,27,30,35,38],\
[2,9,12,17,33,42],\
[10,15,25,29,34,36],\
[2,6,25,26,41,42],\
[6,14,25,27,38,43],\
[7,12,21,33,38,40],\
[2,3,26,33,36,40],\
[5,8,20,31,36,41],\
[5,14,19,20,32,42],\
[8,16,18,22,27,35],\
[11,12,13,17,25,33],\
[12,15,21,27,30,35],\
[6,14,21,22,23,30],\
[5,13,18,22,27,41],\
[6,9,12,13,30,40],\
[4,15,24,40,41,42],\
[2,4,6,15,21,41],\
[1,3,6,7,16,19],\
[11,21,26,31,32,33],\
[6,13,22,28,29,37],\
[8,12,21,22,31,35],\
[3,10,15,18,27,40],\
[13,17,21,25,29,37],\
[8,10,14,15,27,30],\
[5,6,21,27,32,38],\
[5,12,24,25,37,38],\
[4,5,9,20,21,32],\
[10,12,26,28,29,37],\
[14,19,31,35,40,43],\
[7,9,14,30,42,43],\
[1,4,6,14,28,31],\
[2,10,21,28,41,43],\
[2,4,9,36,37,39],\
[3,4,6,18,32,37],\
[7,10,13,23,34,39],\
[22,28,33,36,37,41],\
[2,7,25,26,38,42],\
[10,16,25,29,38,42],\
[5,15,23,26,30,36],\
[4,6,12,18,28,41],\
[1,8,11,16,25,32],\
[24,27,34,35,36,43],\
[5,14,16,22,25,40],\
[3,14,17,32,35,43],\
[11,12,18,25,35,39],\
[4,5,11,13,22,25],\
[2,8,21,27,33,37],\
[4,22,26,27,32,43],\
[6,10,17,18,19,24],\
[3,10,11,13,20,27],\
[1,11,14,21,23,39],\
[5,7,23,24,37,38],\
[5,18,19,26,31,43],\
[2,14,21,26,37,38],\
[1,3,10,21,29,43],\
[5,7,8,12,23,25],\
[19,21,24,28,32,40],\
[4,14,18,26,35,36],\
[5,10,16,34,36,39],\
[4,17,21,28,35,42],\
[14,16,32,35,38,41],\
[1,12,15,17,21,36],\
[3,14,29,32,33,42],\
[1,3,6,17,28,42],\
[10,24,32,34,38,40],\
[10,13,15,25,37,42],\
[4,11,14,18,25,37],\
[2,9,22,26,31,35],\
[3,17,22,23,27,40],\
[5,8,14,20,26,27],\
[2,11,30,32,38,41],\
[1,3,4,10,15,40],\
[6,7,13,15,35,39],\
[9,10,23,31,35,36],\
[9,14,18,28,30,33],\
[2,12,13,22,24,34],\
[2,5,18,27,35,38],\
[3,7,8,26,42,43],\
[8,11,17,26,33,43],\
[8,22,24,28,31,36],\
[3,4,17,24,27,34],\
[1,3,5,12,13,38],\
[12,28,32,36,38,43],\
[5,17,34,37,39,42],\
[7,11,31,34,38,43],\
[5,9,10,28,39,40],\
[4,17,27,39,41,42],\
[5,6,13,20,22,28],\
[19,24,27,31,32,33],\
[3,4,7,13,22,31],\
[5,6,9,18,28,40],\
[2,3,23,32,36,39],\
[2,9,10,24,25,43],\
[2,4,23,24,33,41],\
[5,15,22,24,29,36],\
[6,10,12,16,39,40],\
[16,19,26,27,29,32],\
[5,6,23,30,37,38],\
[6,11,15,24,27,33],\
[7,10,12,34,40,43],\
[1,11,19,28,39,43],\
[10,19,23,25,30,40],\
[5,21,24,34,37,38],\
[16,21,27,32,34,43],\
[5,14,19,24,27,37],\
[1,2,6,24,25,41],\
[4,7,15,23,38,42],\
[9,10,12,13,28,38],\
[1,12,15,26,34,40],\
[2,6,20,24,31,40],\
[3,31,32,38,39,40],\
[2,11,20,31,32,37],\
[13,18,22,28,41,42],\
[1,6,12,20,22,23],\
[3,16,19,25,39,40],\
[2,14,19,22,24,43],\
[6,20,23,25,31,33],\
[9,29,33,34,35,41],\
[9,13,18,19,25,30],\
[2,18,19,20,25,31],\
[8,12,22,23,25,35],\
[4,6,8,10,19,28],\
[1,3,5,25,39,41],\
[1,4,13,18,23,39],\
[1,7,11,14,22,34],\
[13,19,24,28,33,38],\
[15,22,25,29,34,42],\
[10,13,14,19,26,43],\
[1,7,14,25,37,43],\
[1,10,22,24,34,37],\
[9,13,19,27,34,38],\
[12,17,29,32,38,40],\
[2,8,13,27,31,32],\
[4,5,13,21,33,34],\
[10,11,17,31,35,39],\
[1,9,18,20,24,29],\
[1,9,23,29,37,41],\
[9,14,22,24,25,32],\
[19,21,23,24,27,43],\
[5,6,15,23,40,43],\
[6,7,12,26,39,43],\
[10,13,27,34,36,39],\
[5,11,17,24,35,41],\
[15,16,23,29,37,39],\
[2,8,14,23,28,42],\
[1,2,21,22,37,38],\
[2,4,9,13,16,33],\
[4,7,10,11,12,32],\
[7,11,16,19,20,23],\
[15,21,31,32,33,34],\
[17,19,21,22,36,37],\
[1,2,4,16,17,38],\
[1,8,19,25,32,38],\
[2,6,11,27,29,37],\
[6,10,12,32,37,39],\
[4,14,17,22,26,41],\
[6,7,24,34,42,43],\
[6,8,10,21,39,42],\
[6,8,9,20,24,27],\
[8,10,11,12,20,40],\
[3,4,6,15,32,40],\
[14,22,25,28,37,41],\
[11,19,20,25,38,40],\
[4,10,14,19,35,38],\
[8,10,24,26,27,28],\
[10,16,20,25,29,39],\
[16,27,33,35,39,41],\
[2,4,11,15,20,38],\
[3,5,13,18,22,37],\
[13,16,21,25,31,38],\
[4,6,13,16,18,39],\
[7,8,11,13,23,40],\
[2,14,16,27,28,30],\
[8,19,20,23,24,35],\
[6,9,10,29,31,35],\
[8,12,16,23,33,34],\
[2,5,12,14,30,42],\
[3,4,8,10,14,37],\
[2,12,13,14,22,35],\
[8,11,15,29,36,43],\
[8,9,15,26,37,40],\
[11,13,18,20,28,38],\
[14,21,22,32,33,41],\
[6,10,21,24,27,39],\
[2,14,17,19,38,43],\
[5,12,19,22,32,39],\
[10,19,22,27,29,39],\
[5,13,14,26,37,38],\
[20,33,35,36,37,38],\
[6,7,23,25,38,42],\
[10,15,21,22,33,35],\
[2,10,16,18,27,42],\
[10,15,22,23,38,42],\
[3,24,26,27,39,41],\
[13,21,30,31,37,40],\
[5,6,7,20,31,41],\
[4,9,32,37,38,39],\
[16,27,28,36,39,42],\
[6,11,32,34,35,39],\
[1,14,23,24,27,40],\
[25,26,27,28,35,38],\
[11,14,27,31,34,42],\
[15,20,27,29,33,43],\
[13,14,16,21,29,42],\
[5,7,14,24,25,35],\
[5,10,12,25,28,31],\
[2,14,26,28,29,42],\
[19,20,21,24,25,26],\
[2,5,8,9,14,25],\
[5,11,15,24,34,39],\
[4,5,9,20,33,37],\
[1,11,13,29,37,39],\
[2,13,20,22,25,38],\
[8,15,21,32,40,41],\
[2,8,29,31,34,37],\
[14,15,16,18,20,33],\
[7,10,16,18,19,37],\
[20,24,27,28,38,41],\
[1,20,23,37,38,40],\
[20,22,24,32,36,39],\
[1,8,17,24,33,35],\
[12,13,21,25,29,32],\
[5,20,21,26,30,33],\
[2,12,14,18,40,41],\
[2,10,14,19,36,39],\
[4,7,11,12,13,19],\
[8,23,27,31,38,40],\
[4,14,18,22,32,35],\
[9,10,12,19,21,42],\
[4,7,8,15,16,42],\
[7,8,10,21,34,38],\
[6,8,12,17,24,31],\
[9,12,16,33,34,35],\
[16,20,27,33,40,42],\
[21,24,26,35,37,38],\
[12,23,27,32,39,43],\
[6,7,13,18,27,33],\
[16,24,29,32,35,38],\
[3,5,7,18,19,41],\
[1,11,13,30,37,41],\
[8,9,23,34,38,41],\
[4,5,7,37,41,42],\
[8,24,25,29,32,40],\
[4,11,22,31,32,35],\
[2,8,20,33,38,42],\
[8,10,15,21,24,34],\
[8,11,15,22,40,41],\
[2,6,16,27,28,29],\
[1,5,23,24,36,43],\
[5,10,14,16,23,38],\
[1,4,7,25,32,41],\
[4,6,10,17,24,33],\
[7,13,20,27,34,35],\
[21,31,34,37,39,40],\
[2,6,28,30,31,41],\
[13,22,27,29,36,37],\
[2,8,11,12,26,29],\
[13,18,21,30,33,40],\
[12,31,32,34,36,40],\
[6,9,17,20,33,40],\
[8,20,31,36,37,38],\
[21,30,35,39,41,42],\
[14,24,32,40,41,42],\
[3,20,29,33,34,40],\
[2,8,13,17,25,37],\
[5,10,23,26,28,29],\
[12,18,21,29,32,38],\
[13,21,22,26,29,36],\
[1,10,16,19,20,39],\
[4,15,17,34,38,40],\
[12,31,34,40,41,43],\
[14,22,29,33,36,40],\
[22,26,27,28,36,43],\
[2,7,15,22,36,43],\
[9,12,14,24,28,33],\
[3,10,12,32,33,43],\
[4,14,22,23,26,42],\
[2,6,21,31,35,37],\
[8,13,14,21,33,41],\
[3,14,21,24,26,29],\
[3,11,18,20,33,39],\
[1,16,20,24,33,34],\
[6,11,19,28,35,40],\
[7,12,16,28,31,34],\
[2,12,21,23,24,26],\
[8,11,12,15,41,43],\
[8,10,13,15,20,40],\
[7,9,11,14,15,18],\
[3,4,14,21,22,25],\
[2,5,7,10,36,40],\
[3,7,13,18,19,41],\
[6,16,21,30,34,43],\
[7,13,15,18,29,38],\
[2,4,7,19,34,42],\
[6,13,14,22,26,28],\
[5,17,26,37,39,42],\
[4,8,12,22,25,30],\
[7,8,16,35,39,42],\
[7,16,18,20,27,36],\
[6,15,16,28,36,40],\
[7,10,13,19,31,38],\
[2,11,29,31,33,38],\
[4,15,21,31,35,42],\
[5,8,18,28,34,35],\
[4,7,18,19,24,35],\
[6,16,24,32,36,43],\
[2,6,22,25,31,42],\
[2,6,18,26,32,37],\
[7,18,31,32,39,40],\
[10,12,14,24,34,39],\
[2,5,12,23,26,41],\
[18,24,36,37,39,41],\
[4,6,16,17,18,26],\
[2,3,8,12,20,21],\
[8,15,24,26,35,42],\
[1,4,13,25,38,43],\
[9,15,20,22,28,43],\
[3,8,18,21,23,42],\
[8,13,15,18,24,41],\
[6,22,25,26,30,39],\
[3,20,23,24,28,35],\
[1,2,8,13,27,39],\
[6,10,15,21,36,41],\
[3,4,7,13,16,23],\
[2,13,17,24,29,34],\
[1,5,15,22,39,40],\
[4,5,15,30,34,38],\
[2,10,16,26,29,30],\
[12,14,22,23,34,41],\
[1,13,16,22,27,33],\
[20,24,25,27,32,37],\
[1,9,10,28,41,42],\
[8,13,18,33,34,36],\
[10,12,18,27,33,40],\
[6,7,10,23,36,42],\
[1,28,32,34,41,42],\
[10,11,24,26,37,38],\
[2,12,17,18,40,43],\
[7,10,14,25,32,38],\
[5,22,28,29,35,39],\
[12,15,17,25,35,43],\
[6,8,17,19,23,42],\
[16,22,23,27,33,37],\
[3,12,29,31,36,37],\
[8,9,21,24,29,37],\
[4,11,17,21,23,30],\
[4,11,16,20,26,43],\
[3,5,21,30,37,38],\
[10,12,18,38,42,43],\
[6,20,21,23,33,38],\
[2,4,10,28,38,43],\
[4,19,21,25,28,40],\
[7,10,15,31,32,42],\
[7,13,17,23,28,34],\
[10,17,18,29,30,41],\
[3,6,8,21,28,32],\
[3,12,18,24,34,39],\
[4,5,26,28,34,38],\
[9,10,11,25,30,37],\
[4,14,30,35,40,41],\
[4,17,19,37,39,42],\
[2,7,16,25,26,43],\
[4,32,33,38,41,43],\
[6,18,23,32,38,43],\
[6,9,13,17,29,30],\
[2,3,7,9,36,42],\
[16,27,34,35,36,38],\
[3,9,11,15,27,37],\
[13,22,26,30,40,41],\
[15,16,19,21,26,42],\
[12,15,19,32,36,42],\
[5,14,27,29,32,33],\
[12,13,16,28,29,35],\
[6,10,11,30,32,39],\
[1,14,30,34,40,43],\
[16,21,24,28,32,36],\
[11,14,16,32,41,42],\
[15,17,18,19,22,31],\
[10,14,23,35,38,41],\
[3,4,5,28,34,43],\
[15,23,29,35,36,41],\
[3,10,13,17,18,28],\
[2,11,14,23,26,33],\
[10,18,22,24,26,37],\
[4,10,12,14,23,36],\
[6,14,23,24,27,32],\
[1,2,19,22,26,37],\
[6,16,24,31,41,43],\
[23,28,33,35,37,42],\
[5,12,16,24,26,33],\
[1,10,19,30,33,36],\
[6,19,21,24,31,41],\
[2,5,12,21,28,32],\
[5,7,12,23,39,41],\
[8,14,18,19,26,27],\
[1,11,17,28,33,38],\
[7,11,14,18,35,40],\
[7,8,26,27,41,43],\
[3,6,16,23,25,43],\
[1,8,15,21,23,30],\
[2,4,12,14,26,43],\
[1,13,25,29,30,33],\
[14,31,34,35,40,42],\
[6,10,13,14,28,37],\
[2,14,16,17,29,30],\
[2,3,23,25,33,35],\
[6,7,14,15,19,41],\
[1,6,12,14,15,19],\
[14,20,23,25,33,39],\
[4,22,23,27,38,43],\
[12,15,26,35,41,43],\
[3,4,12,18,27,28],\
[1,7,14,15,24,42],\
[5,14,19,21,28,31],\
[5,11,21,27,37,39],\
[19,21,27,33,34,43],\
[12,21,22,27,29,31],\
[1,18,25,27,28,39],\
[9,12,27,28,29,41],\
[3,17,21,23,36,38],\
[6,20,28,31,34,38],\
[3,9,15,16,22,35],\
[1,25,36,41,42,43],\
[6,12,14,17,36,40],\
[2,6,11,29,32,41],\
[4,6,20,21,29,33],\
[8,16,19,23,26,31],\
[8,26,30,36,41,42],\
[1,8,12,20,23,37],\
[7,9,20,27,31,34],\
[5,9,13,28,34,40],\
[6,8,12,30,33,43],\
[5,15,29,34,39,42],\
[2,6,8,19,22,34],\
[16,25,27,29,34,41],\
[14,26,30,33,34,39],\
[13,19,25,29,32,35],\
[2,8,15,24,33,35],\
[4,30,31,36,37,38],\
[2,9,11,24,25,34],\
[2,5,17,18,35,38],\
[5,7,14,17,20,27],\
[11,14,23,30,31,38],\
[12,14,16,18,38,41],\
[7,17,22,30,36,40],\
[5,9,10,27,33,42],\
[12,15,19,25,28,41],\
[6,16,17,22,34,37],\
[10,17,28,30,36,42],\
[2,13,14,22,23,32],\
[1,11,17,23,38,43],\
[10,12,20,35,40,43],\
[9,17,26,30,34,40],\
[18,19,20,24,32,43],\
[1,4,19,28,34,40],\
[6,24,27,30,32,38],\
[10,21,22,25,27,37],\
[10,13,16,25,30,42],\
[2,4,6,16,31,33],\
[8,10,13,30,32,42],\
[18,22,23,32,35,40],\
[11,13,27,31,37,43],\
[1,18,27,30,36,40],\
[10,12,27,32,34,43],\
[3,7,15,25,30,39],\
[11,20,26,28,30,43],\
[2,16,18,27,42,43],\
[7,17,19,27,35,40],\
[1,5,25,27,29,40],\
[6,14,16,19,26,39],\
[15,20,21,32,40,42],\
[19,20,21,23,27,32],\
[9,23,32,35,39,42],\
[15,23,29,36,42,43],\
[3,8,17,26,27,36],\
[10,20,23,26,37,40],\
[5,13,16,18,24,32],\
[6,10,21,22,23,35],\
[2,23,26,35,41,42],\
[10,18,33,34,39,41],\
[9,23,29,30,31,39],\
[11,14,20,28,32,41],\
[3,9,16,17,20,31],\
[2,5,11,21,23,43],\
[4,12,21,22,27,38],\
[2,11,12,28,35,37],\
[3,12,19,33,35,36],\
[3,9,11,24,26,34],\
[8,20,23,24,42,43],\
[1,5,19,20,35,41],\
[1,8,18,26,34,42],\
[17,25,29,33,37,42],\
[8,16,18,35,42,43],\
[5,12,13,33,39,42],\
[6,8,9,29,32,37],\
[2,3,12,20,34,43],\
[3,13,17,25,40,41],\
[1,2,3,30,35,36],\
[5,6,7,8,28,38],\
[2,6,8,10,25,34],\
[14,19,21,28,29,31],\
[5,8,16,25,26,30],\
[3,12,16,25,32,36],\
[3,20,23,30,31,37],\
[12,15,22,24,29,37],\
[9,28,33,34,37,43],\
[1,2,4,5,9,25],\
[2,14,15,26,35,37],\
[2,6,7,9,16,19],\
[2,4,6,24,35,39],\
[2,12,16,20,42,43],\
[6,13,17,27,28,36],\
[12,20,23,28,31,32],\
[11,18,19,30,37,39],\
[1,2,4,16,36,37],\
[6,13,19,20,30,31],\
[3,10,14,18,26,43],\
[8,10,13,20,24,33],\
[3,18,23,24,29,33],\
[14,15,27,34,35,42],\
[4,9,19,22,23,33],\
[1,14,17,28,31,37],\
[2,31,36,37,39,43],\
[15,16,23,30,39,43],\
[9,12,21,26,33,40],\
[15,21,26,28,42,43],\
[14,27,28,31,37,40],\
[7,18,27,37,38,43],\
[10,15,18,30,31,37],\
[8,20,23,24,29,38],\
[5,16,17,22,39,43],\
[6,12,22,25,31,35],\
[5,12,14,18,20,31],\
[12,16,19,27,30,32],\
[2,9,11,15,23,25],\
[6,31,34,35,37,40],\
[17,19,20,22,33,41],\
[7,12,17,18,24,28],\
[2,6,15,28,29,43],\
[4,13,15,22,39,41],\
[17,22,25,26,27,43],\
[8,13,15,21,33,34],\
[11,13,19,26,36,43],\
[2,3,8,20,21,24],\
[20,21,25,27,34,38],\
[7,9,14,17,36,37],\
[4,24,25,26,27,34],\
[7,14,15,19,35,43],\
[7,19,28,33,35,39],\
[14,15,23,24,32,34],\
[2,11,13,15,16,32],\
[3,9,12,16,18,37],\
[7,10,18,21,33,35],\
[10,13,17,38,41,43],\
[12,13,15,18,30,41],\
[6,19,20,28,36,37],\
[7,11,12,19,33,42],\
[3,6,7,22,27,31],\
[7,10,15,20,37,40],\
[1,21,28,30,37,43],\
[2,3,6,14,26,42],\
[3,5,19,20,23,27],\
[12,31,32,34,39,42],\
[2,10,11,12,19,38],\
[1,5,8,12,17,29],\
[1,11,24,26,36,37],\
[4,6,22,31,34,41],\
[1,9,15,16,22,26],\
[6,8,24,25,34,38],\
[3,13,16,24,26,43],\
[13,18,25,26,29,40],\
[8,13,15,19,20,30],\
[4,5,6,27,36,38],\
[8,20,23,27,32,33],\
[4,9,11,12,32,38],\
[10,21,22,25,35,37],\
[8,11,16,20,28,36],\
[13,28,29,37,38,41],\
[8,15,26,27,34,38],\
[4,14,23,25,28,35],\
[4,10,13,31,35,42],\
[3,5,9,26,34,36],\
[8,10,25,31,39,40],\
[5,15,16,20,34,39],\
[3,6,13,17,20,29],\
[4,5,10,23,36,42],\
[11,14,17,18,34,36],\
[1,5,7,15,41,42],\
[7,8,16,24,26,34],\
[2,8,17,24,32,38],\
[5,12,16,19,28,39],\
[6,8,15,17,39,40],\
[3,7,10,12,19,36],\
[6,12,14,28,31,36],\
[1,7,9,14,23,43],\
[6,7,18,29,33,39],\
[5,13,27,37,39,41],\
[9,11,21,23,33,39],\
[18,19,22,25,35,42],\
[21,29,31,36,38,41],\
[1,5,25,27,38,41],\
[15,20,23,28,37,40],\
[2,7,13,16,36,37],\
[7,8,15,30,35,36],\
[3,4,20,30,40,41],\
[1,2,17,24,40,41],\
[3,8,14,30,38,43],\
[1,19,21,31,32,40],\
[2,17,22,29,38,42],\
[1,16,22,36,37,42],\
[3,4,5,18,28,37],\
[1,4,26,36,41,42],\
[3,13,18,20,23,28],\
[1,7,11,32,33,40],\
[1,20,25,28,31,42],\
[1,2,5,23,31,34],\
[3,7,8,15,17,31],\
[4,24,31,35,37,39],\
[2,4,13,23,38,40],\
[4,5,6,9,10,11],\
[4,7,29,33,36,39],\
[2,4,5,17,35,41],\
[12,25,29,31,35,40],\
[6,8,9,20,38,40],\
[4,19,27,30,31,37],\
[9,12,18,20,37,43],\
[2,5,13,14,17,20],\
[6,8,15,17,36,43],\
[7,14,27,31,34,39],\
[10,15,20,28,41,43],\
[11,13,15,17,22,43],\
[7,9,12,17,34,38],\
[3,4,11,25,28,33],\
[1,5,18,34,35,39],\
[10,11,14,28,31,41],\
[1,3,9,14,16,29],\
[20,26,30,33,35,38],\
[6,9,16,22,27,37],\
[1,9,17,31,41,43],\
[3,9,10,11,19,35],\
[6,12,26,37,38,39],\
[7,11,12,23,30,43],\
[9,13,14,17,22,38],\
[7,9,18,20,30,38],\
[6,11,27,30,33,36],\
[4,13,15,31,33,40],\
[4,5,14,27,31,39],\
[4,18,20,21,29,40],\
[7,12,13,20,23,26],\
[9,10,19,25,40,41],\
[17,19,21,29,33,38],\
[1,10,13,15,21,23],\
[3,6,8,14,24,36],\
[3,5,8,22,35,38],\
[11,15,18,24,30,39],\
[2,3,16,22,23,28],\
[2,5,6,8,20,27],\
[1,5,13,22,39,41],\
[5,7,11,19,21,25],\
[6,17,18,21,35,41],\
[11,15,21,31,37,41],\
[1,12,15,23,34,42],\
[3,15,16,17,25,31],\
[7,15,29,30,39,40],\
[15,17,18,28,30,37],\
[6,7,12,17,34,42],\
[12,13,22,23,25,29],\
[15,16,19,22,36,43],\
[2,6,13,22,32,33],\
[2,12,19,25,29,36],\
[3,8,9,13,22,42],\
[1,3,17,23,33,35],\
[11,32,35,36,39,41],\
[8,16,30,32,34,39],\
[16,28,32,34,39,42],\
[3,6,18,24,28,36],\
[11,12,30,36,38,39],\
[5,34,35,38,42,43],\
[1,10,13,22,29,32],\
[8,14,27,33,36,40],\
[11,15,26,31,36,39],\
[4,9,18,30,31,40],\
[5,10,15,20,33,37],\
[1,3,11,13,27,37],\
[1,8,9,14,34,38],\
[6,21,23,24,28,39],\
[11,17,24,26,27,30],\
[6,18,26,40,41,42],\
[1,5,7,26,39,43],\
[1,4,6,16,18,43],\
[4,9,18,31,36,42],\
[8,16,21,24,27,36],\
[5,8,17,26,36,39],\
[6,13,17,29,32,36],\
[8,26,36,40,42,43],\
[9,11,21,31,35,38],\
[12,17,18,19,30,35],\
[7,9,19,26,30,32],\
[4,14,15,22,29,31],\
[8,27,30,31,38,42],\
[23,25,26,29,33,37],\
[27,28,29,34,35,39],\
[8,10,18,22,31,36],\
[8,19,23,27,29,39],\
[9,21,29,33,35,41],\
[1,4,13,18,22,31],\
[4,7,9,11,20,36],\
[4,7,18,26,29,43],\
[1,11,17,32,35,39],\
[10,14,23,26,33,36],\
[1,6,21,22,37,40],\
[8,27,30,35,40,41],\
[18,24,29,32,35,39],\
[17,21,23,30,31,34],\
[1,3,16,22,41,43],\
[4,7,9,16,18,34],\
[2,17,20,25,39,42],\
[12,18,20,27,35,37],\
[9,18,26,28,33,40],\
[10,13,16,17,24,39],\
[4,7,11,19,25,28],\
[3,14,24,25,33,36],\
[4,5,6,17,20,42],\
[9,10,17,22,28,30],\
[4,5,11,16,30,33],\
[5,10,12,22,26,41],\
[4,11,12,21,36,38],\
[3,4,18,31,40,43],\
[3,7,19,27,39,41],\
[4,7,25,27,34,43],\
[17,23,30,31,39,43],\
[3,32,35,36,39,40],\
[3,8,18,25,26,39],\
[12,21,22,27,39,40],\
[7,16,18,33,36,41],\
[8,13,14,18,35,41],\
[5,7,23,31,39,42],\
[14,29,31,36,38,40],\
[22,23,28,37,40,43],\
[1,2,7,21,29,30],\
[4,10,11,12,20,36],\
[3,6,11,13,15,16],\
[10,22,27,30,36,39],\
[6,18,19,30,36,41],\
[8,19,25,31,34,37],\
[2,3,10,22,28,40],\
[2,9,16,36,38,42],\
[6,10,22,30,31,35],\
[10,11,22,23,33,34],\
[5,6,17,23,32,33],\
[10,12,15,17,30,31],\
[3,15,17,19,26,36],\
[1,6,31,32,36,39],\
[10,14,17,19,29,42],\
[14,18,20,28,34,38],\
[9,11,16,21,25,36],\
[1,2,13,14,35,38],\
[2,3,11,17,24,34],\
[3,22,25,30,33,37],\
[23,26,28,33,38,42],\
[4,10,20,30,31,32],\
[12,23,30,31,38,42],\
[1,6,12,35,37,38],\
[5,8,17,19,25,27],\
[6,20,22,29,30,35],\
[5,6,7,18,22,27],\
[7,9,18,26,27,38],\
[4,7,8,12,25,26],\
[4,8,12,16,31,33],\
[3,8,17,25,28,32],\
[10,11,18,20,31,43],\
[8,9,13,20,33,37],\
[3,16,18,20,28,29],\
[16,18,20,29,30,38],\
[7,17,27,36,37,39],\
[1,4,10,21,23,38],\
[1,16,19,33,38,41],\
[17,21,22,27,33,41],\
[2,16,22,27,29,39],\
[8,9,14,24,32,39],\
[17,19,29,37,39,40],\
[10,21,22,28,37,41],\
[7,9,10,16,23,29],\
[10,13,20,22,25,42],\
[3,13,26,31,32,39],\
[7,13,30,33,35,36],\
[2,6,11,14,18,19],\
[17,24,26,35,40,41],\
[3,18,23,31,32,37],\
[6,30,33,34,40,43],\
[22,30,31,34,38,39],\
[6,18,25,26,30,42],\
[6,10,18,20,29,39],\
[14,16,21,26,37,40],\
[5,6,9,10,21,38],\
[11,15,19,30,36,41],\
[4,17,23,30,39,40],\
[14,21,26,28,29,31],\
[19,20,21,25,33,42],\
[3,8,10,22,28,41],\
[15,18,22,25,36,40],\
[3,4,11,12,15,30],\
[9,13,21,30,32,37],\
[5,7,19,20,38,43],\
[8,10,15,16,26,31],\
[8,9,11,27,37,39],\
[7,13,14,29,38,42],\
[3,17,27,29,30,42],\
[16,18,23,25,29,30],\
[11,15,21,31,32,43],\
[5,6,8,19,37,41],\
[2,5,12,18,30,41],\
[9,22,23,25,34,37],\
[4,5,11,12,28,32],\
[8,11,21,22,25,27],\
[1,9,14,22,25,34],\
[1,11,17,26,28,35],\
[1,16,17,27,29,34],\
[4,5,18,27,35,36],\
[21,27,29,30,36,40],\
[6,8,11,14,31,33],\
[5,9,12,28,29,39],\
[7,16,34,37,38,39],\
[4,10,17,34,35,42],\
[1,13,15,34,36,43],\
[2,3,12,15,25,27],\
[11,16,17,18,29,40],\
[10,23,33,36,37,41],\
[5,17,20,24,28,40],\
[13,15,17,26,41,43],\
[13,15,17,18,30,32],\
[3,13,23,30,35,43],\
[5,6,8,20,21,26],\
[3,8,12,36,39,41],\
[3,7,16,24,29,36],\
[5,13,18,22,23,32],\
[17,19,26,28,35,37],\
[1,16,21,28,35,37],\
[10,12,19,25,26,39],\
[1,11,15,23,26,27],\
[1,4,12,16,19,34],\
[20,22,27,32,36,43],\
[5,14,23,32,33,43],\
[5,10,15,28,37,38],\
[1,10,12,16,20,35],\
[2,4,9,19,23,31],\
[3,14,27,28,40,43],\
[1,6,15,19,27,36],\
[10,12,24,29,33,37],\
[1,10,11,13,19,20],\
[6,11,20,23,24,31],\
[1,9,31,35,40,41],\
[16,23,30,35,37,39],\
[1,4,6,11,18,25],\
[7,15,17,30,33,35],\
[2,10,22,23,31,43],\
[2,14,19,22,32,41],\
[11,16,20,28,34,40],\
[2,3,9,12,15,30],\
[4,8,22,25,27,28],\
[20,21,23,28,30,42],\
[9,15,17,30,32,41],\
[4,8,9,13,16,25],\
[2,14,22,27,35,42],\
[10,13,25,31,32,42],\
[11,12,13,14,37,39],\
[14,25,26,31,37,42],\
[2,15,23,37,40,43],\
[1,16,24,25,36,42],\
[7,15,24,37,38,41],\
[5,13,15,25,27,39],\
[2,13,21,31,32,41],\
[2,5,28,30,33,40],\
[3,11,14,25,27,31],\
[11,14,21,24,25,42],\
[7,9,28,32,36,38],\
[3,16,24,31,34,42],\
[1,2,3,15,24,26],\
[4,15,21,24,33,37],\
[18,21,27,33,36,38],\
[5,17,25,34,38,39],\
[10,18,22,25,39,42],\
[2,9,26,27,34,43],\
[3,5,9,17,21,41],\
[1,16,20,23,30,40],\
[3,5,13,30,32,41],\
[2,20,25,30,37,40],\
[8,17,22,25,29,41],\
[3,6,7,23,29,41],\
[7,18,22,25,28,43],\
[4,16,17,19,26,37],\
[8,12,20,35,37,40],\
[6,11,15,20,31,41],\
[4,8,20,30,33,38],\
[3,4,11,12,13,43],\
[11,32,34,37,40,43],\
[15,18,19,30,36,38],\
[7,10,28,32,33,40],\
[8,12,16,19,20,35],\
[5,7,13,19,38,41],\
[7,29,33,35,37,39],\
[4,8,12,30,33,43],\
[6,16,18,27,36,41],\
[14,20,25,34,35,36],\
[5,12,13,29,34,35],\
[4,6,11,20,21,28],\
[13,29,31,37,41,42],\
[14,17,27,28,35,39],\
[12,26,32,37,40,42],\
[1,3,19,21,35,39],\
[11,19,23,38,39,42],\
[11,16,18,20,42,43],\
[7,19,21,23,33,35],\
[6,12,23,25,28,38],\
[9,15,21,23,27,28],\
[16,18,26,27,34,40],\
[1,5,15,31,36,38],\
[1,9,16,20,21,43],\
[2,8,10,13,27,30],\
]

dlists=np.array(dlists)
dlists=dlists[:n_dlists,:]
