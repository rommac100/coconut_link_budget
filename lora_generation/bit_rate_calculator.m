%%%%
% Bitrate Calculator for LoRa
% https://unsigned.io/understanding-lora-parameters/ Using this as some of the basis for the formulas
% Christopher McCormick
%%%%

BW = 500e3;
SF = 7;
CR = 4/5; % CR of 4:5

bit_rate = BW*SF/(2^SF)*CR;
disp(sprintf("bit_rate: %.2f",bit_rate))
