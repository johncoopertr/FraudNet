use crate::network::NeuralNetwork;
use prost::Message;
use std::fs::File;
use std::io::Write;

/// ONNX tensor data types
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum TensorProtoDataType {
    Float = 1,
    Uint8 = 2,
    Int8 = 3,
    Uint16 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    String = 8,
    Bool = 9,
    Float16 = 10,
    Double = 11,
    Uint32 = 12,
    Uint64 = 13,
    Complex64 = 14,
    Complex128 = 15,
}

/// AttributeProto for ONNX
#[derive(Clone, PartialEq, prost::Message)]
pub struct AttributeProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(enumeration = "AttributeType", tag = "20")]
    pub r#type: i32,
    #[prost(float, tag = "2")]
    pub f: f32,
    #[prost(int64, tag = "3")]
    pub i: i64,
    #[prost(bytes, tag = "4")]
    pub s: Vec<u8>,
    #[prost(message, tag = "5")]
    pub t: Option<TensorProto>,
    #[prost(float, repeated, tag = "6")]
    pub floats: Vec<f32>,
    #[prost(int64, repeated, tag = "7")]
    pub ints: Vec<i64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, prost::Enumeration)]
#[repr(i32)]
pub enum AttributeType {
    Float = 1,
    Int = 2,
    String = 3,
    Tensor = 4,
    Floats = 6,
    Ints = 7,
}

/// TensorProto for ONNX
#[derive(Clone, PartialEq, prost::Message)]
pub struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    pub dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    pub data_type: i32,
    #[prost(string, tag = "8")]
    pub name: String,
    #[prost(float, repeated, tag = "4")]
    pub float_data: Vec<f32>,
    #[prost(bytes, tag = "9")]
    pub raw_data: Vec<u8>,
}

/// ValueInfoProto for ONNX
#[derive(Clone, PartialEq, prost::Message)]
pub struct ValueInfoProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(message, tag = "2")]
    pub r#type: Option<TypeProto>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct TypeProto {
    #[prost(message, tag = "1")]
    pub tensor_type: Option<TensorTypeProto>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct TensorTypeProto {
    #[prost(int32, tag = "1")]
    pub elem_type: i32,
    #[prost(message, tag = "2")]
    pub shape: Option<TensorShapeProto>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    pub dim: Vec<TensorShapeDimension>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct TensorShapeDimension {
    #[prost(int64, tag = "1")]
    pub dim_value: i64,
}

/// NodeProto for ONNX
#[derive(Clone, PartialEq, prost::Message)]
pub struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,
    #[prost(string, tag = "3")]
    pub name: String,
    #[prost(string, tag = "4")]
    pub op_type: String,
    #[prost(message, repeated, tag = "5")]
    pub attribute: Vec<AttributeProto>,
}

/// GraphProto for ONNX
#[derive(Clone, PartialEq, prost::Message)]
pub struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<NodeProto>,
    #[prost(string, tag = "2")]
    pub name: String,
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<TensorProto>,
    #[prost(message, repeated, tag = "11")]
    pub input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    pub output: Vec<ValueInfoProto>,
}

/// ModelProto for ONNX
#[derive(Clone, PartialEq, prost::Message)]
pub struct ModelProto {
    #[prost(int64, tag = "1")]
    pub ir_version: i64,
    #[prost(string, repeated, tag = "2")]
    pub opset_import: Vec<String>,
    #[prost(string, tag = "3")]
    pub producer_name: String,
    #[prost(string, tag = "4")]
    pub producer_version: String,
    #[prost(string, tag = "5")]
    pub domain: String,
    #[prost(int64, tag = "6")]
    pub model_version: i64,
    #[prost(message, tag = "7")]
    pub graph: Option<GraphProto>,
}

impl NeuralNetwork {
    /// Export the neural network to ONNX format
    pub fn save_to_onnx(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let model = self.to_onnx_model()?;
        let mut file = File::create(path)?;
        let mut buf = Vec::new();
        model.encode(&mut buf)?;
        file.write_all(&buf)?;
        Ok(())
    }

    fn to_onnx_model(&self) -> Result<ModelProto, Box<dyn std::error::Error>> {
        let mut nodes = Vec::new();
        let mut initializers = Vec::new();
        let mut layer_num = 0;

        // Build the graph layer by layer
        for i in 0..self.weights.len() {
            let is_last_layer = i == self.weights.len() - 1;
            
            // Weight initializer
            let weight_name = format!("weight_{}", i);
            let weight_tensor = TensorProto {
                dims: vec![self.weights[i].rows as i64, self.weights[i].cols as i64],
                data_type: TensorProtoDataType::Float as i32,
                name: weight_name.clone(),
                float_data: self.weights[i].data.iter().map(|&x| x as f32).collect(),
                raw_data: Vec::new(),
            };
            initializers.push(weight_tensor);

            // Bias initializer
            let bias_name = format!("bias_{}", i);
            let bias_tensor = TensorProto {
                dims: vec![self.biases[i].rows as i64],
                data_type: TensorProtoDataType::Float as i32,
                name: bias_name.clone(),
                float_data: self.biases[i].data.iter().map(|&x| x as f32).collect(),
                raw_data: Vec::new(),
            };
            initializers.push(bias_tensor);

            // Input name for this layer
            let input_name = if i == 0 {
                "input".to_string()
            } else {
                format!("activation_{}", i - 1)
            };

            // MatMul node
            let matmul_output = format!("matmul_{}", i);
            let matmul_node = NodeProto {
                input: vec![input_name.clone(), weight_name],
                output: vec![matmul_output.clone()],
                name: format!("MatMul_{}", layer_num),
                op_type: "MatMul".to_string(),
                attribute: Vec::new(),
            };
            nodes.push(matmul_node);
            layer_num += 1;

            // Add node
            let add_output = format!("add_{}", i);
            let add_node = NodeProto {
                input: vec![matmul_output, bias_name],
                output: vec![add_output.clone()],
                name: format!("Add_{}", layer_num),
                op_type: "Add".to_string(),
                attribute: Vec::new(),
            };
            nodes.push(add_node);
            layer_num += 1;

            // Activation node
            let activation_output = if is_last_layer {
                "output".to_string()
            } else {
                format!("activation_{}", i)
            };

            let activation_node = if is_last_layer {
                // Sigmoid for output layer
                NodeProto {
                    input: vec![add_output],
                    output: vec![activation_output],
                    name: format!("Sigmoid_{}", layer_num),
                    op_type: "Sigmoid".to_string(),
                    attribute: Vec::new(),
                }
            } else {
                // ReLU for hidden layers
                NodeProto {
                    input: vec![add_output],
                    output: vec![activation_output],
                    name: format!("Relu_{}", layer_num),
                    op_type: "Relu".to_string(),
                    attribute: Vec::new(),
                }
            };
            nodes.push(activation_node);
            layer_num += 1;
        }

        // Define input
        let input_info = ValueInfoProto {
            name: "input".to_string(),
            r#type: Some(TypeProto {
                tensor_type: Some(TensorTypeProto {
                    elem_type: TensorProtoDataType::Float as i32,
                    shape: Some(TensorShapeProto {
                        dim: vec![
                            TensorShapeDimension { dim_value: 1 },
                            TensorShapeDimension { dim_value: self.layer_sizes[0] as i64 },
                        ],
                    }),
                }),
            }),
        };

        // Define output
        let output_info = ValueInfoProto {
            name: "output".to_string(),
            r#type: Some(TypeProto {
                tensor_type: Some(TensorTypeProto {
                    elem_type: TensorProtoDataType::Float as i32,
                    shape: Some(TensorShapeProto {
                        dim: vec![
                            TensorShapeDimension { dim_value: 1 },
                            TensorShapeDimension {
                                dim_value: self.layer_sizes[self.layer_sizes.len() - 1] as i64,
                            },
                        ],
                    }),
                }),
            }),
        };

        // Create graph
        let graph = GraphProto {
            node: nodes,
            name: "fraudnet".to_string(),
            initializer: initializers,
            input: vec![input_info],
            output: vec![output_info],
        };

        // Create model
        let model = ModelProto {
            ir_version: 8,
            opset_import: vec!["ai.onnx:14".to_string()],
            producer_name: "FraudNet".to_string(),
            producer_version: "0.1.0".to_string(),
            domain: "ai.fraudnet".to_string(),
            model_version: 1,
            graph: Some(graph),
        };

        Ok(model)
    }
}
