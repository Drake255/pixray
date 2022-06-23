import os, sys
import shutil
from pathlib import Path
import pixray
from torch import cuda
import pandas as pd

CWD = Path(__file__).parents[0]

os.chdir(CWD)

print(f'CWD: {os.getcwd()}')

print(f'CUDA Detected: {cuda.is_available()}')

def xls_to_dict_list(path_to_excel:Path) -> list:
    path_to_excel = Path(path_to_excel)
    df = pd.read_excel(path_to_excel).fillna(value='')
    dict_list = df.to_dict('records')
    for d in dict_list:
        for k,v in d.items():
            if k == 'size': #Convert "str" list back to Python list
                d[k] = v.split(',')
                d[k] = [int(i) for i in d[k]]
            if k == 'init_image':
                if v == '':
                    d[k] = None
            if k == 'seed':
                if v == '':
                    d[k] = None
                else:
                    d[k] = int(v)

    return dict_list

def upscale_img(path_to_img:str, model="FSRCNN", factor=2):
    '''
    Upscale output image with OpenCV.
    See: https://learnopencv.com/super-resolution-in-opencv/
    Pretrained models with corresponding factors:
    EDSR:   x2, x3, x4 (best quality but slower)
    ESPCN:  x2, x3, x4
    FSRCNN: x2, x3, x4
    LapSRN: x2, x4, x8

    '''
    import cv2
    from cv2 import dnn_superres

    # Create an SR object
    sr = dnn_superres.DnnSuperResImpl_create()

    # Read image
    image = cv2.imread(path_to_img)

    # Read the desired model
    path = f"./upscale/{model}_x{factor}.pb"
    sr.readModel(path)

    # Set the desired model and scale to get correct pre- and post-processing
    sr.setModel(model.lower(), factor)

    # Upscale the image
    result = sr.upsample(image)

    path_to_img = Path(path_to_img)

    # Save the image
    cv2.imwrite(str(Path(path_to_img.parent, f'{path_to_img.name.split(".")[0]}_x{factor}{path_to_img.suffix}')), result)


def main(dict_list:list, query_name='Queries', output_dir_root='outputs'):

    #Copy the Excel queries file to the output folder
    queries_save_path = Path(CWD, output_dir_root, query_name.split(".")[0], query_name)
    if not queries_save_path.is_dir():
        os.makedirs(queries_save_path.parent, exist_ok=True)

    shutil.copyfile(Path(CWD, query_name), queries_save_path)

    for idx, run in enumerate(dict_list):

        n_runs = int(run['n_runs'])
        run.pop('n_runs')

        upscale_factor = int(run['upscale_factor'])
        run.pop('upscale_factor')

        upscale_model = str(run['upscale_model'])
        run.pop('upscale_model')
        
        output_name = run['output'] #Backup output name for run counting

        ## See: https://dazhizhong.gitbook.io/pixray-docs/docs/primary-settings
        '''
        prompt = 'cat'
        kwargs_for_run = {
            'prompts':prompt,      # Multiple simultaneous prompts can be input, with the pipe/vertical bar character "|" as separation between prompts. For each prompt, it is possible to specify weight and when to stop optimizing for a prompt, with syntax like prompt:weight:stop.
            'quality':'normal',    # draft, normal, better, best, supreme (will manage interations and other params).
            'iterations':None,
            'size':[100,100],      # Size specifies the width * height of the image.
            'aspect':widescreen    # widescreen (default), square
            'drawer':'vqgan',      # vqgan (OK) or pixel (not working) - TODO: vdiff to be installed.
            'init_image':None,     # Path or URL of the initial image that is used to jump start the model. (param is not working)
            'pixel_type':'rect',   # Pixel type can be one of rect, rectshift, tri, diamond, hex, and knit (param is not working)
            'vqgan_model':,        # "imagenet_f16_1024","imagenet_f16_16384","imagenet_f16_16384m","openimages_f16_8192","coco","faceshq","wikiart_1024","wikiart_16384","wikiart_16384m","sflckr"
            'num_cuts':None,       # Reduce VRAM // Specifies the number of "cutouts" that the algorithm feeds into CLIP. A "cutout" is like image augmentation, and allows CLIP to see the image in differing zoom levels, perspectives, and so on. This is much like image augmentation for NN training - the more images, the better the grasp of the model.
            'batches':1,           # Reduce VRAM // The number of batches specifies how many iterations of accumulation of gradients would be needed before the image is altered one step. Think of it the same as in SGD. It can also be increased in conjunction with reducing num_cuts to reduce VRAM needed for image generation.
        }
        '''

        for i in range(n_runs):

            run['output'] = output_name + f'_{i}'
            run['outdir'] = f'{output_dir_root}/{query_name.split(".")[0]}/{output_name}'

            print(f'Performing Pixray Query #[{idx:0>3}] Run #[{i:0>3}]\n   ---> {run}')

            pixray.reset_settings()
            pixray.add_settings(**run)
            settings = pixray.apply_settings()
            pixray.do_init(settings)

            # Save current seed to a txt file
            with open(Path(CWD, run['outdir'], f'seed_{i}.txt'), 'w') as f:
                f.write(str(pixray.global_seed_used))

            pixray.do_run(settings)

            save_path = Path(CWD, run['outdir'], run['output'] + '.png')

            print(f"Run finished and saved to: {save_path}")

            os.rename(Path(CWD, run['outdir'], 'steps'), Path(CWD, run['outdir'], f'steps_{i}')) # Rename the steps folder for each seed of the same query

            if upscale_factor in [2,3,4,8]:
                upscale_img(str(save_path), model=upscale_model, factor=upscale_factor)



if __name__ == '__main__':

    XLS_NAME = 'Queries.xls'
    main(xls_to_dict_list(Path(CWD, XLS_NAME)), XLS_NAME)
    print('All queries were completed without errors!')