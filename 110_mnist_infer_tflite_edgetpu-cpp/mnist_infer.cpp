#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "edgetpu.h"

#include "mnist_infer.h"

using namespace std;
using namespace tflite;

std::unique_ptr<tflite::FlatBufferModel> model;
std::unique_ptr<tflite::Interpreter> interpreter;
ops::builtin::BuiltinOpResolver resolver;
edgetpu::EdgeTpuContext* edgetpu_context;


#if defined (VERBOSE)
static void
print_tensor_dim (int tensor_id)
{
    TfLiteIntArray *dim = interpreter->tensor(tensor_id)->dims;
    
    for (int i = 0; i < dim->size; i ++)
    {
        if (i > 0)
            fprintf (stderr, "x");
        fprintf (stderr, "%d", dim->data[i]);
    }
    fprintf (stderr, "\n");
}

static void
print_tensor_info ()
{
    int i, idx;
    int in_size  = interpreter->inputs().size();
    int out_size = interpreter->outputs().size();

    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    fprintf (stderr, "tensors size     : %zu\n", interpreter->tensors_size());
    fprintf (stderr, "nodes   size     : %zu\n", interpreter->nodes_size());
    fprintf (stderr, "number of inputs : %d\n", in_size);
    fprintf (stderr, "number of outputs: %d\n", out_size);
    fprintf (stderr, "input(0) name    : %s\n", interpreter->GetInputName(0));

    fprintf (stderr, "\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    fprintf (stderr, "                     name                     bytes  type  scale   zero_point\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    int t_size = interpreter->tensors_size();
    for (i = 0; i < t_size; i++) 
    {
        fprintf (stderr, "Tensor[%2d] %-32s %8zu, %2d, %f, %3d\n", i,
            interpreter->tensor(i)->name, 
            interpreter->tensor(i)->bytes,
            interpreter->tensor(i)->type,
            interpreter->tensor(i)->params.scale,
            interpreter->tensor(i)->params.zero_point);
    }

    fprintf (stderr, "\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    fprintf (stderr, " Input Tensor Dimension\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    for (i = 0; i < in_size; i ++)
    {
        idx = interpreter->inputs()[i];
        fprintf (stderr, "Tensor[%2d]: ", idx);
        print_tensor_dim (idx);
    }

    fprintf (stderr, "\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    fprintf (stderr, " Output Tensor Dimension\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    for (i = 0; i < out_size; i ++)
    {
        idx = interpreter->outputs()[i];
        fprintf (stderr, "Tensor[%2d]: ", idx);
        print_tensor_dim (idx);
    }

    fprintf (stderr, "\n");
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
    PrintInterpreterState(interpreter.get());
    fprintf (stderr, "-----------------------------------------------------------------------------\n");
}
#endif


int main(int argc, char* argv[]) 
{
    char tflite_fname_default[] = "../MNIST_models/mnist_quant.tflite";
    char *tflite_fname = tflite_fname_default;

    if (argc > 1)
        tflite_fname = argv[1];

    fprintf (stderr, "-------------------------------------------------------------\n");
    fprintf (stderr, "  TFLite Model: \"%s\"\n", tflite_fname);
    fprintf (stderr, "-------------------------------------------------------------\n");


    model = tflite::FlatBufferModel::BuildFromFile(tflite_fname);
    if (!model)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        exit (-1);
    }

    /* Initialize EdgeTPU context */
    edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->NewEdgeTpuContext().release();
    if (!edgetpu_context)
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);

    resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());

    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter)
    {
        fprintf (stderr, "ERR: %s(%d): Failed to construct interpreter\n", __FILE__, __LINE__);
        exit (-1);
    }

    /* Bind given context with interpreter. */
    interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
    interpreter->SetNumThreads(1);

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d): Failed to allocate tensors\n", __FILE__, __LINE__);
        exit (-1);
    }

#if defined (VERBOSE)
    print_tensor_info ();
#endif

    int  input_index  = interpreter->inputs ()[0];
    int  output_index = interpreter->outputs()[0];
    auto input_type   = interpreter->tensor(input_index )->type;
    auto output_type  = interpreter->tensor(output_index)->type;
    if (input_type != output_type)
    {
        fprintf (stderr, "Type of graph's input (%d) does not match type of its output (%d).\n",
                 int(input_type), int(output_type));
        exit (-1);
    }

    switch (input_type) {
    case kTfLiteFloat32:
        infer_float (interpreter.get());
        break;
    case kTfLiteUInt8:
        infer_uint8 (interpreter.get());
        break;
    default:
         fprintf (stderr, "Unsupported type of graph's input: %d.\n", input_type);
    }

    return 0;
}
