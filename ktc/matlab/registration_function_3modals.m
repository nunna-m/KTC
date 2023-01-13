function reg_img2 = registration_function_3modals(moving_path1,moving_path2,fixed_path)
    moving1 = imread(moving_path1);
    moving2 = imread(moving_path2);
    fixed = imread(fixed_path);
    %register moving1 and fixed first
    [optimizer, metric] = imregconfig ( 'multimodal' );
    optimizer.InitialRadius = optimizer.InitialRadius/10;
    optimizer.MaximumIterations = 300;
    tformSimilarity = imregtform(moving1,fixed,'similarity',optimizer,metric);
    reg_img1 = imregister(moving1,fixed,'affine',optimizer,metric,...
    'InitialTransformation',tformSimilarity);
    %register previously registered img and moving2
    tformSimilarity = imregtform(moving2,reg_img1,'similarity',optimizer,metric);
    reg_img2 = imregister(moving2,reg_img1,'affine',optimizer,metric,...
    'InitialTransformation',tformSimilarity);
    %imshow(reg_img2);
end

% path1 = 'ec.png';
% path2 = 'pc.png';
% path3 = 'tm.png';
% reg_img = registration_function_3modals(path1,path2,path3)