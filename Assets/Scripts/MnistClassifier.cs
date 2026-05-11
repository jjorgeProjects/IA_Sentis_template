using UnityEngine;
using Unity.InferenceEngine;
using TMPro;
using UnityEngine.UI;

public class MnistClassifier : MonoBehaviour
{
    [Header("Assets")]
    [SerializeField] private Texture2D inputTexture;
    [SerializeField] private ModelAsset modelAsset;

    [Header("Debug output")]
    [SerializeField] private int predictedDigit;
    [SerializeField] private float confidence;
    [SerializeField] private float[] results;

    [SerializeField] private TMP_Text resultLabel;

    [SerializeField] private RawImage digitShow;


    private Model runtimeModel;
    private Worker worker;


    private Texture2D prevTexture2D;

    void Start()
    {
        if (inputTexture == null || modelAsset == null)
        {
            Debug.LogError("Asigna inputTexture y modelAsset en el Inspector.");
            return;
        }

        prevTexture2D = inputTexture;

        // 1. Cargar modelo ONNX
        Model sourceModel = ModelLoader.Load(modelAsset);

        // 2. Añadir softmax al output, como hace el ejemplo oficial
        FunctionalGraph graph = new FunctionalGraph();
        FunctionalTensor[] inputs = graph.AddInputs(sourceModel);
        FunctionalTensor[] outputs = Functional.Forward(sourceModel, inputs);
        FunctionalTensor softmax = Functional.Softmax(outputs[0]);
        graph.AddOutput(softmax);
        runtimeModel = graph.Compile();

        // 3. Crear worker
        // GPUCompute suele ir bien; si te da problemas, prueba BackendType.CPU
        worker = new Worker(runtimeModel, BackendType.GPUCompute);

        // 4. Crear tensor de entrada [batch=1, channels=1, height=28, width=28]
        using Tensor<float> inputTensor = new Tensor<float>(new TensorShape(1, 1, 28, 28));


        digitShow.texture = inputTexture;

        // 5. Convertir la textura a tensor
        TextureConverter.ToTensor(inputTexture, inputTensor);

        // 6. Ejecutar inferencia
        worker.Schedule(inputTensor);

        // 7. Leer salida
        Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;
        results = outputTensor.DownloadToArray();

        // 8. Obtener argmax
        predictedDigit = ArgMax(results);
        confidence = results[predictedDigit];

        if (resultLabel != null)
            resultLabel.text = $"Predicción: {predictedDigit} ({confidence:F2})";

        Debug.Log($"Predicción: {predictedDigit} | Confianza: {confidence:F4}");
        Debug.Log("Scores: " + string.Join(", ", results));
    }

    void Predict(){
        if (inputTexture == null || modelAsset == null)
        {
            Debug.LogError("Asigna inputTexture y modelAsset en el Inspector.");
            return;
        }

        prevTexture2D = inputTexture;

        // 1. Cargar modelo ONNX
        Model sourceModel = ModelLoader.Load(modelAsset);

        // 2. Añadir softmax al output, como hace el ejemplo oficial
        FunctionalGraph graph = new FunctionalGraph();
        FunctionalTensor[] inputs = graph.AddInputs(sourceModel);
        FunctionalTensor[] outputs = Functional.Forward(sourceModel, inputs);
        FunctionalTensor softmax = Functional.Softmax(outputs[0]);
        graph.AddOutput(softmax);
        runtimeModel = graph.Compile();

        // 3. Crear worker
        // GPUCompute suele ir bien; si te da problemas, prueba BackendType.CPU
        worker = new Worker(runtimeModel, BackendType.GPUCompute);

        // 4. Crear tensor de entrada [batch=1, channels=1, height=28, width=28]
        using Tensor<float> inputTensor = new Tensor<float>(new TensorShape(1, 1, 28, 28));


        digitShow.texture = inputTexture;

        // 5. Convertir la textura a tensor
        TextureConverter.ToTensor(inputTexture, inputTensor);

        // 6. Ejecutar inferencia
        worker.Schedule(inputTensor);

        // 7. Leer salida
        Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;
        results = outputTensor.DownloadToArray();

        // 8. Obtener argmax
        predictedDigit = ArgMax(results);
        confidence = results[predictedDigit];

        if (resultLabel != null)
            resultLabel.text = $"Predicción: {predictedDigit} ({confidence:F2})";

        Debug.Log($"Predicción: {predictedDigit} | Confianza: {confidence:F4}");
        Debug.Log("Scores: " + string.Join(", ", results));
    }

    private int ArgMax(float[] values)
    {
        if (values == null || values.Length == 0)
            return -1;

        int bestIndex = 0;
        float bestValue = values[0];

        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > bestValue)
            {
                bestValue = values[i];
                bestIndex = i;
            }
        }

        return bestIndex;
    }

    void Update()
    {
        if(!prevTexture2D.Equals(inputTexture))
        {
            Predict();
        }
    }

    void OnDisable()
    {
        worker?.Dispose();
    }
}