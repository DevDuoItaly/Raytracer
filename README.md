# Raytracer

## Overview
This project is a ray tracing renderer implemented in CUDA. It features realistic 3D rendering capabilities using ray tracing algorithms and utilizes the GLM library for mathematical operations.

## Features
- 3D rendering using ray tracing.
- Implementation of various geometric shapes like spheres.
- Camera view and projection handling with GLM library.

## Getting Started
To get started with this project:
1. Clone the repository.
2. Install necessary dependencies, including the GLM library and CUDA toolkit.

### Prerequisites
- CUDA Toolkit
- GLM Library

### Installation
- Detailed steps to set up the project.

## Usage
To run the project, execute the `run.sh` script, which compiles and runs the main CUDA program:
```shell
nvcc ./src/main.cu -I./src/vendor/ -o Raytracer && ./Raytracer
```

## License
This project is licensed under the GNU General Public License (GPL). The GPL is a copyleft license, which means that derivative work must also be open source and distributed under the same terms. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments
- insert credit
