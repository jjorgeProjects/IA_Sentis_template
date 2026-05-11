using UnityEngine;


public class StyleTransferToTexture : MonoBehaviour
{
    public Unity.InferenceEngine.ModelAsset modelAsset;
    public Texture2D inputImage;
    public RenderTexture outputTexture;

    private Unity.InferenceEngine.Worker worker;
    private Unity.InferenceEngine.Tensor<float> inputTensor;

    void Start()
    {
        var sourceModel = Unity.InferenceEngine.ModelLoader.Load(modelAsset);

        var graph = new Unity.InferenceEngine.FunctionalGraph();
        var input = graph.AddInput(sourceModel, 0);
        var output = Unity.InferenceEngine.Functional.Forward(sourceModel, input)[0];

        // Ajuste del ejemplo oficial por si la salida viene en 0..255
        output /= 255f;

        var runtimeModel = graph.Compile(output);
        worker = new Unity.InferenceEngine.Worker(runtimeModel, Unity.InferenceEngine.BackendType.GPUCompute);

        inputTensor = new Unity.InferenceEngine.Tensor<float>(new Unity.InferenceEngine.TensorShape(1, 3, 224, 224));

        RunOnce();
    }

    public void RunOnce()
    {
        Unity.InferenceEngine.TextureConverter.ToTensor(inputImage, inputTensor, new Unity.InferenceEngine.TextureTransform());

        worker.Schedule(inputTensor);
        Unity.InferenceEngine.Tensor<float> outputTensor = worker.PeekOutput() as Unity.InferenceEngine.Tensor<float>;

        Unity.InferenceEngine.TextureConverter.RenderToTexture(outputTensor, outputTexture);
        //Unity.InferenceEngine.TextureConverter.RenderToScreen(outputTensor);
    }

    void OnDisable()
    {
        worker?.Dispose();
        inputTensor?.Dispose();
    }
}