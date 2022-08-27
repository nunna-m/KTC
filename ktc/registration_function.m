function reg_img = registration_function(moving_path,fixed_path)
    moving= imread(moving_path);
    fixed = imread(fixed_path);
    [optimizer, metric] = imregconfig ( 'multimodal' );
    optimizer.InitialRadius = optimizer.InitialRadius/10;
    optimizer.MaximumIterations = 300;
    tformSimilarity = imregtform(moving,fixed,'similarity',optimizer,metric);
    reg_img = imregister(moving,fixed,'affine',optimizer,metric,...
    'InitialTransformation',tformSimilarity);
end

% path1 = 'dc.png';
% path2 = 'pc.png';
% reg_img = registration_function(path1,path2)



