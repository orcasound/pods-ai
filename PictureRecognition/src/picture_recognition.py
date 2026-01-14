from fastai.vision.all import *
from ddgs import DDGS
from fastcore.all import *
from fastdownload import download_url
import time, json

def search_images(keywords, max_images=200):
    return L(DDGS().images(keywords, max_results=max_images)).itemgot('image')

def main():
    print("fastai OK")
    print("ddgs OK")

    # --- Download example images ---
    urls = search_images('orca photos', max_images=1)
    dest = 'orca.jpg'
    download_url(urls[0], dest, show_progress=False)

    im = Image.open(dest)
    im.to_thumb(256,256)

    download_url(search_images('humpback photos', max_images=1)[0], 'humpback.jpg', show_progress=False)
    Image.open('humpback.jpg').to_thumb(256,256)

    download_url(search_images('seal photos', max_images=1)[0], 'seal.jpg', show_progress=False)
    Image.open('seal.jpg').to_thumb(256,256)

    # --- Build dataset ---
    searches = ('orca','humpback','seal')
    path = Path('marine_mammals')

    for o in searches:
        dest = path/o
        dest.mkdir(exist_ok=True, parents=True)
        download_images(dest, urls=search_images(f'{o} photo'))
        time.sleep(5)
        resize_images(dest, max_size=400)

    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)
    print("Failed images:", len(failed))

    # --- DataLoaders ---
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock), 
        get_items=get_image_files, 
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')]
    ).dataloaders(path, bs=32)

    dls.show_batch(max_n=6)

    # --- Train model ---
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(3)

    # --- Predict ---
    is_orca,_,probs = learn.predict(PILImage.create('orca.jpg'))
    print(f"This is a {is_orca}.")
    print(f"Humpback probability: {probs[0]:.4f}")
    print(f"Orca     probability: {probs[1]:.4f}")
    print(f"Seal     probability: {probs[2]:.4f}")

if __name__ == "__main__":
    main()
