function img=PVC_3D(PET,MR,FWHM_x,FWHM_y,FWHM_z,voxsize_x,voxsize_y,voxsize_z,niter,mu)
% 3D PVC algorithms
% PET: input 3D PET image
% MR: input 3D MR image
% FWHM_x: System FWHM in x dimension (unit: mm)
% FWHM_y: System FWHM in y dimension (unit: mm)
% FWHM_z: System FWHM in z dimension (unit: mm)
% voxsize_x: voxel size in x dimension (unit: mm)
% voxsize_y: voxel size in y dimension (unit: mm)
% voxsize_z: voxel size in z dimension (unit: mm)
%% regularization parameters
% mu=17;lambda=5;
%% transform unit of FWHM from mm to grid
FWHM_x=FWHM_x/voxsize_x;
FWHM_y=FWHM_y/voxsize_y;
FWHM_z=FWHM_z/voxsize_z;
%% build 3D PSF
sigma_x=FWHM_x/(2*sqrt(2*log(2)));
sigma_y=FWHM_y/(2*sqrt(2*log(2)));
sigma_z=FWHM_z/(2*sqrt(2*log(2)));
x=1:256;y=1:256;z=1:256;
[X,Y,Z]=meshgrid(x,y,z);
psf=1/((2*pi)^(3/2)*sigma_x*sigma_y*sigma_z)*exp(-((X-128)/sigma_x).^2/2)...
     .*exp(-((Y-128)/sigma_y).^2/2).*exp(-((Z-128)/sigma_z).^2/2);
psf=psf(122:134,122:134,122:134); psf=psf/sum(psf(:));
%% PVC
img=PLS_SplitBregman_3D(PET,MR,psf,niter,mu);
end