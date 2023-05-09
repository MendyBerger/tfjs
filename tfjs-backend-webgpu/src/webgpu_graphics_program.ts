/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

export interface WebGPUGraphicsProgram {
  alphaMode?: boolean;
  // The unique key to distinguish different shader source code.
  shaderKey: string;
  getUserCode: () => string[];
}

export class DrawTextureGraphicsProgram implements WebGPUGraphicsProgram {
  alphaMode = false;
  shaderKey: string;

  constructor(alphaMode: boolean) {
    this.alphaMode = alphaMode;
    this.shaderKey = `drawTexture_${alphaMode}`;
  }

  getUserCode() {
    const vertextShader = `@vertex
    fn main(@builtin(vertex_index) VertexIndex : u32) -> @builtin(position) vec4<f32> {
      var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>( 1.0, -1.0));
      return vec4<f32>(pos[VertexIndex], 0.0, 1.0);
    }`;
    const fragementShader = `
    @group(0) @binding(0) var image0 : texture_2d<f32>;
    @fragment
    fn main(@builtin(position) coord: vec4<f32>) -> @location(0) vec4<f32> {
      var coordVec2 = vec2<i32>(i32(coord.x), i32(coord.y));
      let value = textureLoad(image0, coordVec2, 0);
      return value;
    }`;
    return [vertextShader, fragementShader];
  }
}

export const compileGraphicsProgram =
    (device: GPUDevice, program: WebGPUGraphicsProgram): GPURenderPipeline => {
      const [vertextShader, fragementShader] = program.getUserCode();
      const renderPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: {
          module: device.createShaderModule({
            code: vertextShader,
          }),
          entryPoint: 'main',
        },
        fragment: {
          module: device.createShaderModule({
            code: fragementShader,
          }),
          entryPoint: 'main',
          targets:
              [program.alphaMode ? {
                format: 'bgra8unorm',
                blend: {
                  color: {
                    srcFactor: 'src-alpha',
                    dstFactor: 'one-minus-src-alpha',
                    operation: 'add',
                  },
                  alpha: {
                    srcFactor: 'one',
                    dstFactor: 'one-minus-src-alpha',
                    operation: 'add',
                  },
                },
              } :
                                   {format: 'bgra8unorm'}],

        },
      });
      return renderPipeline;
    };
