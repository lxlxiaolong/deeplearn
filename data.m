clc;clear
close all;

M=12;
annate = [0:1:M-1];
d = 1/2;
N=256;
number = 1;
for i=1:10
    for snr=-15:1:25
        for theta =-60:1:60
            A=exp(-1i*2*pi*d*annate.'*sind(theta));
            S=randn(length(theta),N);
            X=A*S;
            X=awgn(X,snr);
            R=X*X'/N;
            R_up = triu(R);
            R_low = tril(R,-1);
            R_train = real(R_up)+imag(R_low);
            
            
            R_train_norm_A = norm(R_train, 'fro');
            
            % 对矩阵 A 进行 L2 范数归一化
            R_train_guiyi = R_train / R_train_norm_A;
            train(number,1,:,:)= R_train_guiyi;
            number = number+1;
        end
    end
    disp(i)
end