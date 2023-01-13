fid = fopen('/home/maanvi/LAB/github/KidneyTumorClassification/ktc/2ModalitiesFilePathsReduced.txt','rt');
while true
  thisline = fgetl(fid);
  if ~ischar(thisline); break; end  %end of file
    %now check whether the string in thisline is a "word", and store it if it is.
    %then
    paths = split(thisline,',');
    path1 = paths{1};
    path2 = paths{2};
    %reg_img = registration_function(path1,path2);
    reg_img = registration_function(path2,path1);
    disp([path1,path2]);
%     figure(1);
%     imshow(reg_img);
    imwrite(reg_img,paths{3},'png');
 end
 fclose(fid);