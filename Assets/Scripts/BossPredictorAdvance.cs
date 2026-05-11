using System;
using System.Text;
using UnityEngine;
using Unity.InferenceEngine;

public class BossTypePredictorAdvance : MonoBehaviour
{
    [Header("Assets")]
    [SerializeField] private ModelAsset modelAsset;
    [SerializeField] private TextAsset metadataJson;

    [Header("Backend")]
    [SerializeField] private BackendType backendType = BackendType.CPU;

    [Header("Debug / Simulación")]
    [SerializeField] private bool runPredictionOnStart = false;
    [SerializeField] private bool showDebugOverlay = true;
    [SerializeField] private KeyCode predictKey = KeyCode.P;

    [SerializeField] private PlayerBossFeatures simulatedFeatures = new PlayerBossFeatures
    {
        avg_session_minutes = 48f,
        aim_accuracy = 0.82f,
        dodge_success_rate = 0.61f,
        aggression_score = 0.87f,
        exploration_score = 0.22f,
        healing_usage_per_min = 0.35f,
        deaths_per_hour = 1.8f,
        melee_ratio = 0.78f,
        ranged_ratio = 0.20f,
        parry_ratio = 0.25f,
        boss_fail_streak = 1f,
        upgrade_power_score = 0.72f
    };

    [Header("Último resultado")]
    [SerializeField] private string lastPredictedBoss = "-";
    [SerializeField] private string lastProbabilities = "-";

    private Model runtimeModel;
    private Worker worker;
    private BossModelMetadata metadata;

    public bool IsReady => worker != null && metadata != null;

    private void Awake()
    {
        Initialize();
    }

    private void Start()
    {
        if (runPredictionOnStart)
            PredictBossWithSimulatedData();
    }

    private void Update()
    {
        if (Input.GetKeyDown(predictKey))
            PredictBossWithSimulatedData();
    }

    public void Initialize()
    {
        if (worker != null)
            return;

        if (modelAsset == null)
        {
            Debug.LogError("BossTypePredictor: falta asignar modelAsset.");
            return;
        }

        if (metadataJson == null)
        {
            Debug.LogError("BossTypePredictor: falta asignar metadataJson.");
            return;
        }

        metadata = JsonUtility.FromJson<BossModelMetadata>(metadataJson.text);
        if (metadata == null || metadata.feature_columns == null || metadata.class_names == null)
        {
            Debug.LogError("BossTypePredictor: metadata JSON inválido.");
            return;
        }

        runtimeModel = ModelLoader.Load(modelAsset);
        worker = new Worker(runtimeModel, backendType);
    }

    [ContextMenu("Predict Boss With Simulated Data")]
    public void PredictBossWithSimulatedData()
    {
        try
        {
            BossType predicted = PredictBossType(simulatedFeatures, out float[] probabilities);

            lastPredictedBoss = predicted.ToString();
            lastProbabilities = FormatProbabilities(probabilities);

            Debug.Log($"[BossTypePredictor] Predicción: {lastPredictedBoss}");
            Debug.Log($"[BossTypePredictor] Probabilidades:\\n{lastProbabilities}");
        }
        catch (Exception ex)
        {
            lastPredictedBoss = "ERROR";
            lastProbabilities = ex.Message;
            Debug.LogException(ex);
        }
    }

    public BossType PredictBossType(PlayerBossFeatures features)
    {
        return PredictBossType(features, out _);
    }

    public BossType PredictBossType(PlayerBossFeatures features, out float[] probabilities)
    {
        probabilities = null;

        if (!IsReady)
        {
            Initialize();
            if (!IsReady)
                throw new InvalidOperationException("BossTypePredictor no está inicializado.");
        }

        int featureCount = metadata.feature_columns.Length;

        using Tensor<float> inputTensor = new Tensor<float>(new TensorShape(1, featureCount));

        for (int i = 0; i < featureCount; i++)
        {
            float value = GetFeatureValueByName(features, metadata.feature_columns[i]);
            inputTensor[0, i] = value;
        }

        worker.Schedule(inputTensor);

        Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;
        if (outputTensor == null)
            throw new InvalidOperationException("La salida del modelo no es Tensor<float>.");

        using Tensor<float> cpuTensor = outputTensor.ReadbackAndClone() as Tensor<float>;
        if (cpuTensor == null)
            throw new InvalidOperationException("No se pudo clonar la salida a CPU.");

        float[] logits = new float[metadata.class_names.Length];
        for (int i = 0; i < logits.Length; i++)
            logits[i] = cpuTensor[0, i];

        probabilities = Softmax(logits);

        int bestIndex = ArgMax(probabilities);
        string className = metadata.class_names[bestIndex];

        if (!Enum.TryParse(className, out BossType bossType))
        {
            throw new InvalidOperationException(
                $"No se pudo convertir la clase '{className}' a BossType."
            );
        }

        return bossType;
    }

    private int ArgMax(float[] values)
    {
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

    private float[] Softmax(float[] logits)
    {
        float max = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > max) max = logits[i];

        float sum = 0f;
        float[] probs = new float[logits.Length];

        for (int i = 0; i < logits.Length; i++)
        {
            probs[i] = Mathf.Exp(logits[i] - max);
            sum += probs[i];
        }

        if (sum > 0f)
        {
            for (int i = 0; i < probs.Length; i++)
                probs[i] /= sum;
        }

        return probs;
    }

    private string FormatProbabilities(float[] probabilities)
    {
        var sb = new StringBuilder();

        for (int i = 0; i < probabilities.Length; i++)
        {
            sb.Append(metadata.class_names[i]);
            sb.Append(": ");
            sb.Append(probabilities[i].ToString("F4"));

            if (i < probabilities.Length - 1)
                sb.AppendLine();
        }

        return sb.ToString();
    }

    private float GetFeatureValueByName(PlayerBossFeatures f, string featureName)
    {
        switch (featureName)
        {
            case "avg_session_minutes": return f.avg_session_minutes;
            case "aim_accuracy": return f.aim_accuracy;
            case "dodge_success_rate": return f.dodge_success_rate;
            case "aggression_score": return f.aggression_score;
            case "exploration_score": return f.exploration_score;
            case "healing_usage_per_min": return f.healing_usage_per_min;
            case "deaths_per_hour": return f.deaths_per_hour;
            case "melee_ratio": return f.melee_ratio;
            case "ranged_ratio": return f.ranged_ratio;
            case "parry_ratio": return f.parry_ratio;
            case "boss_fail_streak": return f.boss_fail_streak;
            case "upgrade_power_score": return f.upgrade_power_score;
            default:
                throw new ArgumentOutOfRangeException(nameof(featureName), $"Feature no reconocida: {featureName}");
        }
    }

    private void OnGUI()
    {
        if (!showDebugOverlay)
            return;

        const int x = 12;
        int y = 12;
        const int width = 420;
        const int lineHeight = 22;

        GUI.Box(new Rect(x, y, width, 240), "BossTypePredictor Debug");
        y += 30;

        GUI.Label(new Rect(x + 10, y, width - 20, lineHeight), $"Ready: {IsReady}");
        y += lineHeight;

        GUI.Label(new Rect(x + 10, y, width - 20, lineHeight), $"Predicted Boss: {lastPredictedBoss}");
        y += lineHeight + 4;

        GUI.Label(new Rect(x + 10, y, width - 20, lineHeight), "Press P to predict with simulated data");
        y += lineHeight + 6;

        GUI.Label(new Rect(x + 10, y, width - 20, 120), lastProbabilities ?? "-");
    }

    private void OnDestroy()
    {
        worker?.Dispose();
        worker = null;
    }
}