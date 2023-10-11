"""
 Stitching sample (advanced)
 ===========================

 Show how to use Stitcher API from python.
 """

 # Python 2/3 compatibility
 from __future__ import print_function

 import numpy as np
 import cv2 as cv

 import sys
 import argparse
 #创建parser实例
 parser = argparse.ArgumentParser(prog='stitching_detailed.py', description='Rotation model images stitcher')
 #继续增加参数
 parser.add_argument('img_names', nargs='+',help='files to stitch',type=str)
 #两个必须的入参，执行文件名，图片名
 parser.add_argument('--preview',help='Run stitching in the preview mode. Works faster than usual mode but output image will have lower resolution.',type=bool,dest = 'preview' )
 parser.add_argument('--try_cuda',action = 'store', default = False,help='Try to use CUDA. The default value is no. All default values are for CPU mode.',type=bool,dest = 'try_cuda' )
 #默认不用cuda
 #存储值到变量，action = store
 parser.add_argument('--work_megapix',action = 'store', default = 0.6,help=' Resolution for image registration step. The default is 0.6 Mpx',type=float,dest = 'work_megapix' )
 parser.add_argument('--features',action = 'store', default = 'orb',help='Type of features used for images matching. The default is orb.',type=str,dest = 'features' )
 parser.add_argument('--matcher',action = 'store', default = 'homography',help='Matcher used for pairwise image matching.',type=str,dest = 'matcher' )
 parser.add_argument('--estimator',action = 'store', default = 'homography',help='Type of estimator used for transformation estimation.',type=str,dest = 'estimator' )
 parser.add_argument('--match_conf',action = 'store', default = 0.3,help='Confidence for feature matching step. The default is 0.65 for surf and 0.3 for orb.',type=float,dest = 'match_conf' )
 parser.add_argument('--conf_thresh',action = 'store', default = 1.0,help='Threshold for two images are from the same panorama confidence.The default is 1.0.',type=float,dest = 'conf_thresh' )
 parser.add_argument('--ba',action = 'store', default = 'ray',help='Bundle adjustment cost function. The default is ray.',type=str,dest = 'ba' )
 parser.add_argument('--ba_refine_mask',action = 'store', default = 'xxxxx',help='Set refinement mask for bundle adjustment.  mask is "xxxxx"',type=str,dest = 'ba_refine_mask' )
 parser.add_argument('--wave_correct',action = 'store', default = 'horiz',help='Perform wave effect correction. The default is "horiz"',type=str,dest = 'wave_correct' )
 parser.add_argument('--save_graph',action = 'store', default = None,help='Save matches graph represented in DOT language to <file_name> file.',type=str,dest = 'save_graph' )
 parser.add_argument('--warp',action = 'store', default = 'plane',help='Warp surface type. The default is "spherical".',type=str,dest = 'warp' )
 parser.add_argument('--seam_megapix',action = 'store', default = 0.1,help=' Resolution for seam estimation step. The default is 0.1 Mpx.',type=float,dest = 'seam_megapix' )
 #缝合线尺度0.1M大概10万像素，320*320
 parser.add_argument('--seam',action = 'store', default = 'no',help='Seam estimation method. The default is "gc_color".',type=str,dest = 'seam' )
 parser.add_argument('--compose_megapix',action = 'store', default = -1,help='Resolution for compositing step. Use -1 for original resolution.',type=float,dest = 'compose_megapix' )
 parser.add_argument('--expos_comp',action = 'store', default = 'no',help='Exposure compensation method. The default is "gain_blocks".',type=str,dest = 'expos_comp' )
 parser.add_argument('--expos_comp_nr_feeds',action = 'store', default = 1,help='Number of exposure compensation feed.',type=np.int32,dest = 'expos_comp_nr_feeds' )
 parser.add_argument('--expos_comp_nr_filtering',action = 'store', default = 2,help='Number of filtering iterations of the exposure compensation gains',type=float,dest = 'expos_comp_nr_filtering' )
 parser.add_argument('--expos_comp_block_size',action = 'store', default = 32,help='BLock size in pixels used by the exposure compensator.',type=np.int32,dest = 'expos_comp_block_size' )
 parser.add_argument('--blend',action = 'store', default = 'multiband',help='Blending method. The default is "multiband".',type=str,dest = 'blend' )
 parser.add_argument('--blend_strength',action = 'store', default = 5,help='Blending strength from [0,100] range.',type=np.int32,dest = 'blend_strength' )
 parser.add_argument('--output',action = 'store', default = 'result.jpg',help='The default is "result.jpg"',type=str,dest = 'output' )
 parser.add_argument('--timelapse',action = 'store', default = None,help='Output warped images separately as frames of a time lapse movie, with "fixed_" prepended to input file names.',type=str,dest = 'timelapse' )
 parser.add_argument('--rangewidth',action = 'store', default = -1,help='uses range_width to limit number of images to match with.',type=int,dest = 'rangewidth' )

 __doc__ += '\n' + parser.format_help()
 # __doc__可以用函数名调用，会输出函数下的"""xxxxxx """三引号注释中的内容

 def main():
     args = parser.parse_args()
     #参数解析
     #参数应用
     img_names=args.img_names
 #这种传参方法才是合理的，因为服务器传来的params是一个整体的json文件。因此参数解析要用类字典的形式。来的是整体，即便只有一个参数也不能直接类似params = 1如此传参
     print(img_names)
     preview = args.preview
     try_cuda = args.try_cuda
     work_megapix = args.work_megapix
     seam_megapix = args.seam_megapix
     compose_megapix = args.compose_megapix
     conf_thresh = args.conf_thresh
     features_type = args.features
     matcher_type = args.matcher
     estimator_type = args.estimator
     ba_cost_func = args.ba
     ba_refine_mask = args.ba_refine_mask
     wave_correct = args.wave_correct
     if wave_correct=='no':
         do_wave_correct= False
     else:
         do_wave_correct=True
     if args.save_graph is None:
         save_graph = False
     else:
         save_graph =True
         save_graph_to = args.save_graph
     warp_type = args.warp
     #曝光补偿供4种方式：增量补偿，块增量补偿，通道补偿，块通道补偿
     if args.expos_comp=='no':
         expos_comp_type = cv.detail.ExposureCompensator_NO
     elif  args.expos_comp=='gain':
         expos_comp_type = cv.detail.ExposureCompensator_GAIN
     elif  args.expos_comp=='gain_blocks':
         expos_comp_type = cv.detail.ExposureCompensator_GAIN_BLOCKS
     elif  args.expos_comp=='channel':
         expos_comp_type = cv.detail.ExposureCompensator_CHANNELS
     elif  args.expos_comp=='channel_blocks':
         expos_comp_type = cv.detail.ExposureCompensator_CHANNELS_BLOCKS
     else:
         print("Bad exposure compensation method")
         exit()
     expos_comp_nr_feeds = args.expos_comp_nr_feeds
     expos_comp_nr_filtering = args.expos_comp_nr_filtering
     expos_comp_block_size = args.expos_comp_block_size
     match_conf = args.match_conf
     seam_find_type = args.seam
     blend_type = args.blend
     blend_strength = args.blend_strength
     result_name = args.output
     if args.timelapse is not None:
         timelapse = True
         if args.timelapse=="as_is":
             timelapse_type = cv.detail.Timelapser_AS_IS
         elif args.timelapse=="crop":
             timelapse_type = cv.detail.Timelapser_CROP
         else:
             print("Bad timelapse method")
             exit()
     else:
         timelapse= False
     range_width = args.rangewidth
     #特征描述方法，三种方式 orb                surf      sift, 参数解析默认值是orb
     if features_type=='orb':
         finder= cv.ORB.create()
     elif features_type=='surf':
         finder= cv.xfeatures2d_SURF.create()
     elif features_type=='sift':
         finder= cv.xfeatures2d_SIFT.create()
     else:
         print ("Unknown descriptor type")
         exit()
     seam_work_aspect = 1
     full_img_sizes=[]
     features=[]
     images=[]
     is_work_scale_set = False
     is_seam_scale_set = False
     is_compose_scale_set = False;
     for name in img_names:#循环 读图放到list中
         full_img = cv.imread(cv.samples.findFile(name))

         if full_img is None:
             print("Cannot read image ", name)
             exit()
         full_img_sizes.append((full_img.shape[1],full_img.shape[0]))
         if work_megapix < 0:
             img = full_img
             work_scale = 1
             is_work_scale_set = True
         else:
             if is_work_scale_set is False:
                 work_scale = min(1.0, np.sqrt(work_megapix * 1e6 / (full_img.shape[0]*full_img.shape[1])))
                 is_work_scale_set = True
             img = cv.resize(src=full_img, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)
         if is_seam_scale_set is False:
             seam_scale = min(1.0, np.sqrt(seam_megapix * 1e6 / (full_img.shape[0]*full_img.shape[1])))
             seam_work_aspect = seam_scale / work_scale
             #缝合区域大小/配准
             is_seam_scale_set = True
         imgFea= cv.detail.computeImageFeatures2(finder,img)
         #单张图片的特征提取，finder是特征提取器对象
         features.append(imgFea)
         #根据以上来调整图片大小，保存到images图片列表中
         img = cv.resize(src=full_img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv.INTER_LINEAR_EXACT)
         images.append(img)
     if matcher_type== "affine":
     #匹配类型  仿射
         matcher = cv.detail_AffineBestOf2NearestMatcher(False, try_cuda, match_conf)
     elif range_width==-1:
         matcher = cv.detail.BestOf2NearestMatcher_create(try_cuda, match_conf)
     else:
         matcher = cv.detail.BestOf2NearestRangeMatcher_create(range_width, try_cuda, match_conf)
     #此时的features是所有入参图片的特征集合
     p=matcher.apply2(features)
     matcher.collectGarbage()
     if save_graph:
         f = open(save_graph_to,"w")
         f.write(cv.detail.matchesGraphAsString(img_names, p, conf_thresh))
         f.close()
     indices=cv.detail.leaveBiggestComponent(features,p,0.3)
     img_subset =[]
     img_names_subset=[]
     full_img_sizes_subset=[]
     num_images=len(indices)
     for i in range(len(indices)):
         img_names_subset.append(img_names[indices[i,0]])
         img_subset.append(images[indices[i,0]])
         full_img_sizes_subset.append(full_img_sizes[indices[i,0]])
     images = img_subset;
     img_names = img_names_subset;
     full_img_sizes = full_img_sizes_subset;
     num_images = len(img_names)
     if num_images < 2:
         print("Need more images")
         exit()

     if estimator_type == "affine":
         estimator = cv.detail_AffineBasedEstimator()#基于仿射变换特征
     else:
         estimator = cv.detail_HomographyBasedEstimator()#基于透视变换的单应阵
     b, cameras =estimator.apply(features,p,None)#变换估计出变换矩阵和相机参数
     if not b:
         print("Homography estimation failed.")
         exit()
     for cam in cameras:
     #相机的旋转矩阵
         cam.R=cam.R.astype(np.float32)

     if ba_cost_func == "reproj":
         adjuster = cv.detail_BundleAdjusterReproj()
     elif ba_cost_func == "ray":#默认，光束平差的损失函数类型
         adjuster = cv.detail_BundleAdjusterRay()
     elif ba_cost_func == "affine":
         adjuster = cv.detail_BundleAdjusterAffinePartial()
     elif ba_cost_func == "no":
         adjuster = cv.detail_NoBundleAdjuster()
     else:
         print( "Unknown bundle adjustment cost function: ", ba_cost_func )
         exit()
     adjuster.setConfThresh(1)#确定两幅图片属于同一张全景图的阈值
     refine_mask=np.zeros((3,3),np.uint8)
     #掩膜，3*3的0阵，如果要优化则将对应位置赋成1
 """数字图像处理中,掩模为二维矩阵数组,有时也用多值图像。数字图像处理中,图像掩模主要用于：
 
 ①提取感兴趣区,用预先制作的感兴趣区掩模与待处理图像相乘,得到感兴趣区图像,感兴趣区内图像值保持不变,而区外图像值都为0。 
 ②屏蔽作用,用掩模对图像上某些区域作屏蔽,使其不参加处理或不参加处理参数的计算,或仅对屏蔽区作处理或统计。 
 ③结构特征提取,用相似性变量或图像匹配方法检测和提取图像中与掩模相似的结构特征。 
 """

     if ba_refine_mask[0] == 'x':
         refine_mask[0,0] = 1
     if ba_refine_mask[1] == 'x':
         refine_mask[0,1] = 1
     if ba_refine_mask[2] == 'x':
         refine_mask[0,2] = 1
     if ba_refine_mask[3] == 'x':
         refine_mask[1,1] = 1
     if ba_refine_mask[4] == 'x':
         refine_mask[1,2] = 1
     adjuster.setRefinementMask(refine_mask)
     b,cameras = adjuster.apply(features,p,cameras)
     #光束平差，精修相机姿态等
 #光束是指相机光心发出的射线。损失函数一般有重映射误差和射线距离误差两种
     if not b:
         print("Camera parameters adjusting failed.")
         exit()
     focals=[]
     for cam in cameras:
         focals.append(cam.focal) #焦距
     sorted(focals)
     #可以通过两张图的匹配点或单应阵求出焦距，用所有焦距的平均作为全景图焦距的近似。用这个焦距作为柱面的半径，先将图像投射到柱面，在柱面上区拼接
     if len(focals)%2==1:
     #根据焦距确定 图片投影的尺度 warped_image_scale
         warped_image_scale = focals[len(focals) // 2]
     else:
         warped_image_scale = (focals[len(focals) // 2]+focals[len(focals) // 2-1])/2
     if do_wave_correct:
     #波形修正得到修正后的旋转矩阵
         rmats=[]
         for cam in cameras:
             rmats.append(np.copy(cam.R))
         rmats    =    cv.detail.waveCorrect(    rmats,  cv.detail.WAVE_CORRECT_HORIZ)
         for idx,cam in enumerate(cameras):
             cam.R = rmats[idx]
     corners=[]
     mask=[]
     masks_warped=[]
     images_warped=[]
     sizes=[]
     masks=[]
     for i in range(0,num_images):
         um=cv.UMat(255*np.ones((images[i].shape[0],images[i].shape[1]),np.uint8))
         masks.append(um)
     #处理相机旋转造成的扭曲
     warper = cv.PyRotationWarper(warp_type,warped_image_scale*seam_work_aspect) # warper peut etre nullptr?
     for idx in range(0,num_images):
         K = cameras[idx].K().astype(np.float32)#相机内参
         swa = seam_work_aspect
         K[0,0] *= swa
         K[0,2] *= swa
         K[1,1] *= swa
         K[1,2] *= swa
         corner,image_wp =warper.warp(images[idx],K,cameras[idx].R,cv.INTER_LINEAR, cv.BORDER_REFLECT)
         corners.append(corner)
         sizes.append((image_wp.shape[1],image_wp.shape[0]))
         images_warped.append(image_wp)

         p,mask_wp =warper.warp(masks[idx],K,cameras[idx].R,cv.INTER_NEAREST, cv.BORDER_CONSTANT)#线性差值，连续边界处理
         masks_warped.append(mask_wp.get())
     images_warped_f=[]
     for img in images_warped:#对投影后的图片，单精度存储
         imgf=img.astype(np.float32)
         images_warped_f.append(imgf)
     if cv.detail.ExposureCompensator_CHANNELS == expos_comp_type:
         compensator = cv.detail_ChannelsCompensator(expos_comp_nr_feeds)
     #    compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
     elif cv.detail.ExposureCompensator_CHANNELS_BLOCKS == expos_comp_type:
         compensator=cv.detail_BlocksChannelsCompensator(expos_comp_block_size, expos_comp_block_size,expos_comp_nr_feeds)
     #    compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
     else:
         compensator=cv.detail.ExposureCompensator_createDefault(expos_comp_type)
         #曝光补偿需要        分块 投影图片 和 掩膜
 #对于分块补偿，得到分块曝光补偿系数后，此时直接补偿会出现明显的视觉分"块"效果还要对系数进行分段线性滤波处理，得到全局最终的补偿系数
     compensator.feed(corners=corners, images=images_warped, masks=masks_warped)
     #缝合线（两图最佳的一条融合曲线）查找描述子
 #一般有三种方法：1基于距离的逐点法  2.动态规划查找法（快，省内存）   3.最大流图割法
     if seam_find_type == "no":
         seam_finder = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_NO)
     elif seam_find_type == "voronoi":
         seam_finder = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_VORONOI_SEAM);
     elif seam_find_type == "gc_color":
         seam_finder = cv.detail_GraphCutSeamFinder("COST_COLOR")
     elif seam_find_type == "gc_colorgrad":
         seam_finder = cv.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
     elif seam_find_type == "dp_color":
         seam_finder = cv.detail_DpSeamFinder("COLOR")
     elif seam_find_type == "dp_colorgrad":
         seam_finder = cv.detail_DpSeamFinder("COLOR_GRAD")
     if seam_finder is None:
         print("Can't create the following seam finder ",seam_find_type)
         exit()
     seam_finder.find(images_warped_f, corners,masks_warped )
     imgListe=[]
     compose_scale=1
     corners=[]
     sizes=[]
     images_warped=[]
     images_warped_f=[]
     masks=[]
     blender= None
     timelapser=None
     compose_work_aspect=1
     for idx,name in enumerate(img_names): # https://github.com/opencv/opencv/blob/master/samples/cpp/stitching_detailed.cpp#L725 ?
         full_img  = cv.imread(name)
         if not is_compose_scale_set:
             if compose_megapix > 0:
             #exp是2.718; e是10^
                 compose_scale = min(1.0, np.sqrt(compose_megapix * 1e6 / (full_img.shape[0]*full_img.shape[1])))
             is_compose_scale_set = True;
             compose_work_aspect = compose_scale / work_scale;
             warped_image_scale *= compose_work_aspect
             warper =  cv.PyRotationWarper(warp_type,warped_image_scale)
             for i in range(0,len(img_names)):
                 cameras[i].focal *= compose_work_aspect
                 cameras[i].ppx *= compose_work_aspect
                 cameras[i].ppy *= compose_work_aspect
                 sz = (full_img_sizes[i][0] * compose_scale,full_img_sizes[i][1]* compose_scale)
                 K = cameras[i].K().astype(np.float32)
                 roi = warper.warpRoi(sz, K, cameras[i].R);
                 #缝合区获得
                 corners.append(roi[0:2])
                 sizes.append(roi[2:4])
         if abs(compose_scale - 1) > 1e-1:
             img =cv.resize(src=full_img, dsize=None, fx=compose_scale, fy=compose_scale, interpolation=cv.INTER_LINEAR_EXACT)
         else:
             img = full_img;
         img_size = (img.shape[1],img.shape[0]);
         K=cameras[idx].K().astype(np.float32)
         corner,image_warped =warper.warp(img,K,cameras[idx].R,cv.INTER_LINEAR, cv.BORDER_REFLECT)
         mask =255*np.ones((img.shape[0],img.shape[1]),np.uint8)
         p,mask_warped =warper.warp(mask,K,cameras[idx].R,cv.INTER_NEAREST, cv.BORDER_CONSTANT)
         #块增益补偿
         compensator.apply(idx,corners[idx],image_warped,mask_warped)
         image_warped_s = image_warped.astype(np.int16)
         image_warped=[]
         #膨胀
         dilated_mask = cv.dilate(masks_warped[idx],None)
         seam_mask = cv.resize(dilated_mask,(mask_warped.shape[1],mask_warped.shape[0]),0,0,cv.INTER_LINEAR_EXACT)
         mask_warped = cv.bitwise_and(seam_mask,mask_warped)#二进制的与运算，1&1 == 1    ，1&0 == 0
         if blender==None and not timelapse:
             blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
             dst_sz = cv.detail.resultRoi(corners=corners,sizes=sizes)
             blend_width = np.sqrt(dst_sz[2]*dst_sz[3]) * blend_strength / 100
             if blend_width < 1:
                 blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
             elif blend_type == "multiband":
                 blender = cv.detail_MultiBandBlender()
                 blender.setNumBands((np.log(blend_width)/np.log(2.) - 1.).astype(np.int))
             elif blend_type == "feather":#羽化融合
                 blender = cv.detail_FeatherBlender()
                 blender.setSharpness(1./blend_width)
             blender.prepare(dst_sz)
         elif timelapser==None  and timelapse:
             timelapser = cv.detail.Timelapser_createDefault(timelapse_type)
             timelapser.initialize(corners, sizes)
         if timelapse:
         #延时处理
             matones=np.ones((image_warped_s.shape[0],image_warped_s.shape[1]), np.uint8)
             timelapser.process(image_warped_s, matones, corners[idx])
             pos_s = img_names[idx].rfind("/");
             if pos_s == -1:
                 fixedFileName = "fixed_" + img_names[idx];
             else:
                 fixedFileName = img_names[idx][:pos_s + 1 ]+"fixed_" + img_names[idx][pos_s + 1: ]
             cv.imwrite(fixedFileName, timelapser.getDst())
         else:
             blender.feed(cv.UMat(image_warped_s), mask_warped, corners[idx])
     if not timelapse:
         result=None
         result_mask=None
         result,result_mask = blender.blend(result,result_mask)
         cv.imwrite(result_name,result)
         zoomx = 600.0 / result.shape[1]
         dst=cv.normalize(src=result,dst=None,alpha=255.,norm_type=cv.NORM_MINMAX,dtype=cv.CV_8U)
         dst=cv.resize(dst,dsize=None,fx=zoomx,fy=zoomx)
         cv.imshow(result_name,dst)
         cv.waitKey()

     print('Done')


 if __name__ == '__main__':
     print(__doc__)
     main()
     cv.destroyAllWindows()
