clear all;
[moving,map_am] = imread('pc.png');
[fixed,map_pc] = imread("dc.png");
% figure(1);
% imshow(fixed, map_pc);
% title ( 'Unregistered' )

% moving = imread('pc.png');
% fixed = imread('dc.png');

figure(1);
imshow(fixed, map_pc);
title ( 'Unregistered' )

[optimizer, metric] = imregconfig ( 'multimodal' );

movingRegisteredDefault = imregister (moving, fixed, 'affine' , optimizer, metric);

figure(2);
imshowpair(movingRegisteredDefault,fixed,'blend','Scaling','joint')
title('A: Default Registration')

optimizer.InitialRadius = optimizer.InitialRadius/3.5;
% movingRegisteredAdjustedInitialRadius = imregister(moving, fixed, 'affine', optimizer, metric);
% 
% figure(3);
% imshowpair(movingRegisteredAdjustedInitialRadius, fixed)
% title('B: Adjusted Initial Radius')
% 
optimizer.MaximumIterations = 300;
movingRegisteredAdjustedInitialRadius300 = imregister(moving, fixed,'affine',optimizer,metric);

figure(4);
imshowpair(movingRegisteredAdjustedInitialRadius300, fixed,'blend','Scaling','joint')
title('C: Adjusted InitialRadius, MaximumIterations=300')
% 
% figure(5);
tformSimilarity = imregtform(moving,fixed,'similarity',optimizer,metric);
%Rfixed = imref2d(size(fixed));
%movingRegisteredRigid = imwarp(moving,tformSimilarity,'OutputView',Rfixed);
% imshowpair(movingRegisteredRigid, fixed)
% title('D: Registration Baed on Similarity Transformation Model')

figure(6);
movingRegisteredAffineWithIC = imregister(moving,fixed,'affine',optimizer,metric,...
    'InitialTransformation',tformSimilarity);
imshowpair(movingRegisteredAffineWithIC,fixed,'blend','Scaling','joint')
title('E: Registraion from Affine Model Based on Similarity Initial Condition')


another = imread('tm.png')





