function out_fname=PVC_3D_wrapper(pet_fname, mr_fname, out_fname,FWHM_x,FWHM_y,FWHM_z,niter,mu)
% MCR use char input, convert to double
FWHM_x=str2double(FWHM_x);FWHM_y=str2double(FWHM_y);FWHM_z=str2double(FWHM_z);

if nargin<7
    niter=100;
else
    niter=str2double(niter);
end
if nargin<8
    mu=17;
else
    mu=str2double(mu);
end
% load images
% nifti read/write toolbox:
% https://www.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image
% addpath('../nifti_toolbox');
% addpath('./modified_niftiload/');
pet_img = load_untouch_nii(pet_fname);
mr_img = load_untouch_nii(mr_fname);

PET = double(pet_img.img);
MR = double(mr_img.img);

voxsize_x = pet_img.hdr.dime.pixdim(2);
voxsize_y = pet_img.hdr.dime.pixdim(3);
voxsize_z = pet_img.hdr.dime.pixdim(4);

PET_PVC = PVC_3D(PET,MR,FWHM_x,FWHM_y,FWHM_z,voxsize_x,voxsize_y,voxsize_z,niter,mu);

% save output
pet_img.hdr.dime.datatype=64; pet_img.hdr.dime.bitpix=64; % save as double precision
pet_img.img = PET_PVC;
save_untouch_nii(pet_img, out_fname);

end