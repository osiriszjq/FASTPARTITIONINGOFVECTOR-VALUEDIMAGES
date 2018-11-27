clear
im = im2double(imread('desert.jpg'));
%im=round(im,4);
subplot(231)
imagesc(im)
title('Original')
%im = im + 0.1*randn(size(im)); % Noise
% subplot(332)
% imagesc(im)
% title('Noisy')
% c0=inf;

tic
u = ADMM4V(im,1.55,2,0.01*1.55);
t1 = toc;
u=round(u,3);
c1=cost(im,u,2);
subplot(232)
imagesc(u)
title(['ADMM4 Down-Sampling time:',num2str(t1)])
tic
u = ADMM4Vd(im,1.8,2,0.018);
t2 = toc;
u=round(u,3);
c2=cost(im,u,2);
subplot(232)
imagesc(u)
title(['ADMM4 time:',num2str(t2)])

% tic
% [h2, img] = ADMM48Vd_zjq(im,2.6,2,0.026);
% t3 = toc;
% img=round(img,3);
% c3=cost(im,img,2);
% subplot(236)
% imagesc(img);
% title(['ADMM8 Down-sampling time:',num2str(t3)])
% tic
% [~,h3] = ADMM8V_zjq(im,2,2,0.02);
% t4 = toc;
% h3=round(h3,3);
% c4=cost(im,h3,2);
% subplot(233)
% imagesc(h3);
% title(['ADMM8 time:',num2str(t3)])

