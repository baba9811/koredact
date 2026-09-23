//! Execution-provider selection; every accelerator attempt has a fresh session builder.
use std::path::Path;
use ort::session::Session;
use crate::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Device { Cpu, Cuda, Rocm, DirectMl, OpenVinoGpu, Qnn, OpenVinoNpu, DirectMlNpu, CoreMl, CoreMlGpu, CoreMlNpu }

impl Device {
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu", Self::Cuda => "cuda", Self::Rocm => "rocm", Self::DirectMl => "directml",
            Self::OpenVinoGpu => "openvino_gpu", Self::Qnn => "qnn", Self::OpenVinoNpu => "openvino_npu",
            Self::DirectMlNpu => "directml_npu", Self::CoreMl => "coreml", Self::CoreMlGpu => "coreml_gpu", Self::CoreMlNpu => "coreml_npu",
        }
    }
}

pub(crate) fn choices(preference: &str) -> Result<&'static [Device], Error> {
    use Device::*;
    Ok(match preference {
        "auto" => &[Cuda, Rocm, DirectMl, OpenVinoGpu, Qnn, OpenVinoNpu, DirectMlNpu, CoreMl, Cpu],
        "cpu" => &[Cpu],
        "gpu" => &[Cuda, Rocm, DirectMl, OpenVinoGpu, CoreMlGpu, Cpu],
        "npu" => &[Qnn, OpenVinoNpu, DirectMlNpu, CoreMlNpu, Cpu],
        "cuda" => &[Cuda, Cpu], "rocm" => &[Rocm, Cpu], "directml" => &[DirectMl, Cpu],
        "openvino_gpu" => &[OpenVinoGpu, Cpu], "qnn" => &[Qnn, Cpu], "openvino_npu" => &[OpenVinoNpu, Cpu],
        "directml_npu" => &[DirectMlNpu, Cpu], "coreml" => &[CoreMl, Cpu],
        "metal" | "mps" | "coreml_gpu" => &[CoreMlGpu, Cpu], "coreml_npu" => &[CoreMlNpu, Cpu],
        _ => return Err(Error::Bundle(format!("unknown device {preference:?}; use auto, cpu, gpu, npu, cuda, rocm, directml, openvino_gpu, qnn, openvino_npu, directml_npu, coreml, metal, mps, coreml_gpu or coreml_npu"))),
    })
}

pub(crate) fn select<T>(preference: &str, mut load: impl FnMut(Device) -> Result<T, Error>) -> Result<(T, Device, Option<String>), Error> {
    let mut failures = Vec::new();
    for &device in choices(preference)? {
        match load(device) {
            Ok(value) => return Ok((value, device, if failures.is_empty() { None } else { Some(failures.join("; ")) })),
            Err(error) if device != Device::Cpu => failures.push(format!("{}: {error}", device.name())),
            Err(error) => return Err(error),
        }
    }
    unreachable!("every device preference ends in CPU")
}

pub(crate) fn session(onnx: &Path, threads: usize, device: Device) -> Result<Session, Error> {
    let options = |error| Error::Bundle(format!("session options: {error}"));
    let mut builder = Session::builder()?
        .with_no_environment_execution_providers().map_err(options)?
        .with_intra_threads(threads).map_err(options)?
        .with_inter_threads(1).map_err(options)?
        .with_config_entry("session.intra_op.allow_spinning", "0").map_err(options)?;
    if device != Device::Cpu {
        #[cfg(feature = "load-dynamic")]
        {
            use ort::ep::{self, ExecutionProvider};
            use ep::coreml::{ComputeUnits, ModelFormat};
            let provider: Box<dyn ExecutionProvider> = match device {
                Device::Cuda => Box::new(ep::CUDA::default().with_tf32(false)),
                Device::Rocm => Box::new(ep::ROCm::default()),
                Device::DirectMl | Device::DirectMlNpu => {
                    builder = builder.with_memory_pattern(false).map_err(options)?;
                    Box::new(ep::DirectML::default().with_device_filter(if device == Device::DirectMl { ep::directml::DeviceFilter::Gpu } else { ep::directml::DeviceFilter::Npu }))
                }
                Device::OpenVinoGpu | Device::OpenVinoNpu => Box::new(ep::OpenVINO::default().with_device_type(if device == Device::OpenVinoGpu { "GPU" } else { "NPU" }).with_num_threads(threads)),
                Device::Qnn => Box::new(ep::QNN::default().with_backend_path(if cfg!(target_os = "windows") { "QnnHtp.dll" } else { "libQnnHtp.so" })),
                Device::CoreMl | Device::CoreMlGpu | Device::CoreMlNpu => Box::new(ep::CoreML::default()
                    .with_model_format(ModelFormat::MLProgram)
                    .with_compute_units(match device { Device::CoreMlGpu => ComputeUnits::CPUAndGPU, Device::CoreMlNpu => ComputeUnits::CPUAndNeuralEngine, _ => ComputeUnits::All })
                    .with_low_precision_accumulation_on_gpu(false)),
                Device::Cpu => unreachable!(),
            };
            if !provider.is_available()? { return Err(Error::Bundle("provider unavailable in loaded ONNX Runtime".into())); }
            provider.register(&mut builder)?;
        }
        #[cfg(not(feature = "load-dynamic"))]
        return Err(Error::Bundle("accelerators require load-dynamic".into()));
    }
    Ok(builder.commit_from_file(onnx)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unavailable_and_failed_accelerators_fall_back_without_hiding_cpu_errors() {
        let mut attempts = Vec::new();
        let (value, device, reason) = select("gpu", |d| {
            attempts.push(d);
            if d == Device::Cpu { Ok(42) } else { Err(Error::Bundle("unavailable or session creation failed".into())) }
        }).unwrap();
        assert_eq!(value, 42); assert_eq!(device, Device::Cpu);
        assert_eq!(attempts, choices("gpu").unwrap()); assert!(reason.unwrap().contains("session creation failed"));
        assert!(select::<()>("auto", |_| Err(Error::Bundle("CPU also failed".into()))).unwrap_err().to_string().contains("CPU also failed"));
    }
    #[test]
    fn success_stops_selection_and_explicit_cpu_never_probes_accelerators() {
        let mut attempts = Vec::new();
        let (_, device, reason) = select("auto", |d| { attempts.push(d); Ok(()) }).unwrap();
        assert_eq!(device, Device::Cuda); assert_eq!(attempts, [Device::Cuda]); assert!(reason.is_none());
        assert_eq!(choices("cpu").unwrap(), [Device::Cpu]);
        assert_eq!(choices("mps").unwrap(), choices("metal").unwrap());
        assert!(choices("invalid").is_err());
    }
}
