using System;

[System.Serializable]
public class BossModelMetadata
{
    public string[] feature_columns;
    public string[] class_names;
    public int[] input_shape;
    public string target_column;
}