# Caffe

This fork allows to use more MemoryData layers.

## Example

Example of net configuration, that uses two memory data layers. Memory data layer is strict about the dimensionality of label output. In this example, label can have arbitrary dimensions.

```
layer {
  name: "data"
  type: "MemoryData"
  top: "data"
  top: "data_foo"
  include {
    phase: TRAIN
  }
  memory_data_param {
    batch_size: 8
    channels: 3
    height: 227
    width: 227
  }
}
layer {
  name: "label"
  type: "MemoryData"
  top: "label"
  top: "label_foo"
  include {
    phase: TRAIN
  }
  memory_data_param {
    batch_size: 8
    channels: 1844
    height: 1
    width: 1
  }
}
layer {
  name: "silence1"
  type: "Silence"
  bottom: "label_foo"
  include {
    phase: TRAIN
  }
}
layer {
  name: "silence2"
  type: "Silence"
  bottom: "data_foo"
  include {
    phase: TRAIN
  }
}
```

Unnecessary output (*_foo) is silenced with Silence layers. MemoryData layer has always two outputs, so we cannot omit ony of them.

In python, the usage is like this:

```python
solver=caffe.SGDSolver(....)
solver.net.set_input_arrays(data, label_foo, 0)
solver.net.set_input_arrays(label, label_foo, 1)
solver.step(....)
```

set_input_arrays method has additional parameter, index of memory data layer, into which the arrays will be connected