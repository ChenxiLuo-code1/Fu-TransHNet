from PIL import Image
import glob
import os



out_dir = 'D:/wyy/polyp/TransFuse-resnet/TestData/CVC-EndoSceneStill/masks11'
cnt = 0
for img in glob.glob('D:/wyy/polyp/TransFuse-resnet/TestData/CVC-EndoSceneStill/CVC-612/gtpolyp/*.tif'):
    Image.open(img).save(os.path.join(out_dir, str(cnt) + '.png'))
    cnt += 1
