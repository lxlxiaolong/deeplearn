clc;clear
close all;

M=12;
annate = [0:1:M-1];
d = 1/2;
N=256;
number = 0;
number1 = 1;
train_label=[];
for i=1:10
    for snr=-15:1:25
        number=0;
        for theta =-60:1:60
                train_label(number1) =number; 
                number=number+1;
                number1 = number1+1;
        end
    end
    disp(i)
end
train_label = train_label';