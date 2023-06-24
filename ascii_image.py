from re import ASCII
from PIL import Image
ASCII_CHARS = [".",",",":","*","+","^"]
ASCII_CHARS= ASCII_CHARS[::1]

def resize(image,new_width=100):
    (old_width,old_height)= image.size 
    aspect_ratio = float(old_height)/float(old_width)
    new_height =int(aspect_ratio*new_width)
    new_dimension= (new_width,new_height)
    new_image=image.resize(new_dimension)
    return new_image

def grayscale(image):
    return image.convert("L")


def modify(image,bucket=25):
    initial_pixels = list(image.getdata())
    new_pixels = [ASCII_CHARS[pixel_value]for pixel_value in initial_pixels]
    return "".join(new_pixels)


def do(image,new_width=100):
    image=resize(image)
    image =grayscale(image)

    pixels = modify(image)
    len_pixels = len(pixels)
    new_image = [pixels[index:index+new_width] for index in range(0,len_pixels,new_width)]
    return "\n".join(new_image)


def start(path):
    image=None
    try:
        image=Image.oprn(path)
    except Exception:
        print("Unable to find image")
        return

    image=do(image)
    print(image)
    f=open("image.txt","w")
    f.write(image)
    f.close()

if __name__=="__main__":
    import sys
    import urllib.request
    if sys.argv[1].startswith("http://") or sys.argv[1].startswith("http://"):
        urllib.request.urlretrieve(sys.argv[1],"")
        path = ""
    else:
        path=sys.argv[1]
    
    start(path)

