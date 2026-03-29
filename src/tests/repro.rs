#[cfg(test)]
mod reproduction_test {
    use crate::{
        MlResult,
        nn::Parameter,
        tensor::{ComputationGraph, Tensor, TENSOR_STORAGE, TensorBase},
        var_input
    };

    #[test]
    #[cfg(all(feature = "enableBackward"))]
    fn test_graph_reset_drops_variable_data() -> MlResult<()> {
        let _ = crate::tests::common::logging::setup_logging();

        // 1. Create a variable
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = Tensor::from_vec(data.clone(), &shape)?;
        let id = tensor.id();
        let var = var_input!(tensor);

        println!("Variable created. ID: {:?}", id);

        // Verify data exists
        let exists = TENSOR_STORAGE.with(|s| s.borrow().contains_key(&id));
        assert!(exists, "Data should exist initially");

        // 2. Add to graph (simulate what happens in training)
        // We need to simulate with_grad_fn or manual addition
        var.with_grad_fn("test_op", &[]);

        println!("Added to graph.");

        // 3. Reset graph
        ComputationGraph::reset_graph();
        println!("Graph reset.");

        // 4. Check if data still exists
        let exists_after = TENSOR_STORAGE.with(|s| s.borrow().contains_key(&id));

        if !exists_after {
            println!("CRITICAL: Data released after reset_graph!");
        } else {
            println!("Data still exists.");
        }

        assert!(exists_after, "Data should persist because 'var' holds a reference");

        Ok(())
    }

    #[test]
    
    fn test_tensor_with_id_destruction() -> MlResult<()> {
        let _ = crate::tests::common::logging::setup_logging();

        // 1. Create a tensor
        let data = vec![1.0, 2.0];
        let shape = vec![1, 2];
        let tensor = Tensor::from_vec(data.clone(), &shape)?;
        let id = tensor.id();

        println!("Tensor created. ID: {:?}", id);

        // 2. Create a temporary tensor with SAME ID using Tensor::with_id
        // This simulates what happens in assign_forward or accumulate_grad
        {
            let temp = Tensor::with_id(vec![3.0, 4.0], &shape, id)?;
            println!("Temp tensor created with same ID.");
            // temp drops here
        }
        println!("Temp tensor dropped.");

        // 3. Check if original tensor data exists
        let exists = TENSOR_STORAGE.with(|s| s.borrow().contains_key(&id));

        if !exists {
            println!("CRITICAL: Data released after dropping temp tensor created with with_id!");
        } else {
            println!("Data still exists.");
        }

        assert!(exists, "Data should persist because original tensor holds a reference");
        Ok(())
    }
}
