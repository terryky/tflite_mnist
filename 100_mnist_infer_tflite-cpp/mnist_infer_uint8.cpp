#include <iostream>
#include <fstream>
#include <thread>

#if defined (TF_LITE_1_13) || defined (TF_LITE_1_14)
  #include "tensorflow/lite/kernels/register.h"
  #include "tensorflow/lite/model.h"
  #include "tensorflow/lite/version.h"
  #include "tensorflow/lite/optional_debug_tools.h"
#else
  #include "tensorflow/contrib/lite/kernels/register.h"
  #include "tensorflow/contrib/lite/model.h"
  #include "tensorflow/contrib/lite/optional_debug_tools.h"
#endif

#include "mnist_infer.h"

//#define VERBOSE 1

static uint64_t
get_time_us()
{
   struct timespec tv;
   uint64_t us;

   clock_gettime(CLOCK_MONOTONIC, &tv);

   us  = tv.tv_sec  * 1000000ULL;
   us += tv.tv_nsec / 1000;

   return us;
}

static void
swap_read_4b (std::ifstream &file, char *buf)
{
    char readbuf[4];

    file.read (readbuf, 4);
    buf[0] = readbuf[3];
    buf[1] = readbuf[2];
    buf[2] = readbuf[1];
    buf[3] = readbuf[0];
}


static char *
load_mnist_images (int *num_imgs)
{
    std::string path = "../MNIST_data/t10k-images-idx3-ubyte";
    int magic, num_images, rows, cols;
    char *img_buf;

    std::ifstream file;
    file.open (path, std::ios::in | std::ios::binary);
    if (!file)
    {
        fprintf (stderr, "can't open %s\n", path.c_str());
        return 0;
    }

    swap_read_4b (file, (char *)&magic);
    swap_read_4b (file, (char *)&num_images);
    swap_read_4b (file, (char *)&rows);
    swap_read_4b (file, (char *)&cols);
#if defined (VERBOSE)
    fprintf (stderr, "input image: (%d x %d) x %d\n", rows, cols, num_images);
#endif

    int buf_size = num_images * rows * cols;
    img_buf = new char [buf_size];
    file.read(img_buf, buf_size);

    *num_imgs = num_images;
    return img_buf;
}

static char *
load_mnist_labels (int *num_imgs)
{
    std::string path = "../MNIST_data/t10k-labels-idx1-ubyte";
    int magic;
    int num_images;
    char *label_buf;

    std::ifstream file;
    file.open (path, std::ios::in | std::ios::binary);
    if (!file)
    {
        fprintf (stderr, "can't open %s\n", path.c_str());
        return 0;
    }

    swap_read_4b (file, (char *)&magic);
    swap_read_4b (file, (char *)&num_images);

    label_buf = new char [num_images];
    file.read(label_buf, num_images);

    *num_imgs = num_images;
    return label_buf;
}


static void *
get_next_image (char *img_buf, int idx, uint8_t *dst_img)
{
    int x, y;
    int w = 28;
    int h = 28;

    img_buf += (w * h) * idx;

    for (y = 0; y < h; y ++)
    {
        for (x = 0; x < w; x ++)
        {
            uint8_t pix8 = img_buf[y * w + x];
#if defined (VERBOSE)
            int pix = pix8;
            char pict[] = " .:-=+*#%@";
            fprintf (stderr, "%c", pict[pix/26]);
#endif
            *dst_img = pix8;
            dst_img ++;
        }
#if defined (VERBOSE)
        fprintf (stderr, "\n");
#endif
    }
    return 0;
}

static char
get_next_label (char *label_buf, int idx)
{
    return label_buf[idx];
}

static char
argmax (uint8_t *args, int numarg)
{
    float maxval = 0;
    char  maxidx = 0;

    for (int i = 0; i < numarg; i ++)
    {
        if (args[i] > maxval)
        {
            maxval = args[i];
            maxidx = i;
        }
    }
    
    return maxidx;
}



int 
infer_uint8 (tflite::Interpreter *interpreter)
{
    int num_imgs = 0;
    char *image_buf = load_mnist_images (&num_imgs);
    char *label_buf = load_mnist_labels (&num_imgs);


    uint8_t *in_tensor_ptr  = interpreter->typed_input_tensor<uint8_t>(0);
    uint8_t *out_tensor_ptr = interpreter->typed_output_tensor<uint8_t>(0);
    
    uint64_t begin_time = get_time_us ();
    
    int num_correct = 0;
    for (int idx = 0; idx < num_imgs; idx ++)
    {
        get_next_image (image_buf, idx, in_tensor_ptr);
        int label = get_next_label (label_buf, idx);

        if (interpreter->Invoke() != kTfLiteOk)
        {
            fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
            return -1;
        }

        char result = argmax (out_tensor_ptr, 10);
        if (result == label)
            num_correct ++;

#if defined (VERBOSE)
        for (int i = 0; i < 10; i ++)
            fprintf (stderr, "%d ", out_tensor_ptr[i]);

        fprintf (stderr, "\n");
        fprintf (stderr, "result: %d\n", result);
        fprintf (stderr, "label : %d\n", label);
#endif
    }

    uint64_t proc_us = get_time_us () - begin_time;
    float    proc_ms = (float)proc_us / 1000.0f;

    float accuracy = (float)num_correct / (float)num_imgs;
    fprintf (stderr, "\n");
    fprintf (stderr, "----------------------------------------\n");
    fprintf (stderr, "  accuracy(quant) = %f [%d/%d]\n",  accuracy, num_correct, num_imgs);
    fprintf (stderr, "----------------------------------------\n");
    fprintf (stderr, "  proc total   time = %f[ms]\n", proc_ms);
    fprintf (stderr, "  proc average time = %f[ms]\n", proc_ms/num_imgs);
    fprintf (stderr, "----------------------------------------\n");
    fprintf (stderr, "\n");

    return 0;
}
    
    
    