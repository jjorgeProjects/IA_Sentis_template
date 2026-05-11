using UnityEngine;
using Unity.InferenceEngine;

public class BossTypePredictor : MonoBehaviour
{
    [SerializeField] private ModelAsset modelAsset;

    private Worker worker;
    private Model runtimeModel;

    private readonly string[] classNames =
    {
        "Berserker",
        "Juggernaut",
        "Sniper",
        "Summoner",
        "Trickster"
    };

    void Start()
    {
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = new Worker(runtimeModel, BackendType.CPU);

        float avg_session_minutes = 48f;
        float aim_accuracy = 0.82f;
        float dodge_success_rate = 0.61f;
        float aggression_score = 0.87f;
        float exploration_score = 0.22f;
        float healing_usage_per_min = 0.35f;
        float deaths_per_hour = 1.8f;
        float melee_ratio = 0.78f;
        float ranged_ratio = 0.20f;
        float parry_ratio = 0.25f;
        float boss_fail_streak = 1f;
        float upgrade_power_score = 0.72f;
        
        float[] features = {
            avg_session_minutes,
            aim_accuracy,
            dodge_success_rate,
            aggression_score,
            exploration_score,
            healing_usage_per_min,
            deaths_per_hour,
            melee_ratio,
            ranged_ratio,
            parry_ratio,
            boss_fail_streak,
            upgrade_power_score
        };

        Debug.Log($"Predicted: {PredictBossType(features)}");
    }

    public string PredictBossType(float[] features)
    {
        // features debe tener longitud 12
        using Tensor<float> inputTensor = new Tensor<float>(new TensorShape(1, 12));

        for (int i = 0; i < 12; i++)
            inputTensor[0, i] = features[i];

        worker.Schedule(inputTensor);
        Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;

        float[] logits = outputTensor.DownloadToArray();

        int bestIndex = 0;
        float bestValue = logits[0];

        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > bestValue)
            {
                bestValue = logits[i];
                bestIndex = i;
            }
        }

        return classNames[bestIndex];
    }

    void OnDestroy()
    {
        worker?.Dispose();
    }
}