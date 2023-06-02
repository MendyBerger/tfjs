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
import * as tf from '@tensorflow/tfjs-core';

import {describeWebGPU} from './test_util';

const {expectArraysEqual, expectArraysClose} = tf.test_util;

function getCanvas() {
  return document.createElement('canvas');
}

function removeLastChannel(data: Uint8ClampedArray) {
  const pixels = [];
  for (let i = 0; i < data.length; i += 4) {
    pixels.push(data[i], data[i + 1], data[i + 2]);
  }
  return pixels;
}

function unmultiplyAlpha(data: Uint8ClampedArray, dataType: string) {
  const pixels = [];
  const MAX_COLOR = 255;
  for (let i = 0; i < data.length; i += 4) {
    const alpha =
        dataType === 'float32' ? data[i + 3] / MAX_COLOR : data[i + 3];
    pixels.push(
        data[i] / MAX_COLOR * alpha, data[i + 1] / MAX_COLOR * alpha,
        data[i + 2] / MAX_COLOR * alpha, alpha);
  }
  return pixels;
}

async function readPixelsFromGPUCanvas(
    webgpuCanvas: ImageBitmapSource, width: number, height: number,
    postProcess: (data: Uint8ClampedArray, dataType: string) => number[] = null,
    dataType = 'float32') {
  const snapshot = await createImageBitmap(webgpuCanvas);
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(snapshot, 0, 0);
  const imageData = new Uint8ClampedArray(
      ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height).data);
  if (postProcess) {
    return postProcess(imageData, dataType);
  } else {
    return imageData;
  }
}

describeWebGPU('draw on webgpu context', (env) => {
  beforeAll(async () => {
    await tf.setBackend(env.name);
  });

  it('draw image with 4 channels and int values', async () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    const width = 2;
    const height = 2;
    const img = tf.tensor3d(data, [width, height, 4], 'int32');
    const canvas = getCanvas();
    const startNumTensors = tf.memory().numTensors;
    tf.browser.draw(
        // tslint:disable-next-line:no-any
        img, canvas as any, {contextOptions: {contextType: env.name}});
    expect(tf.memory().numTensors).toEqual(startNumTensors);
    expectArraysClose(
        await readPixelsFromGPUCanvas(
            canvas, height, width, unmultiplyAlpha, 'int32'),
        data, 0.1);
  });

  it('draw image with 4 channels and float values', async () => {
    // Premultiplied data.
    const data =
        [.1, .2, .3, .4, .5, .6, .7, .8, .09, .1, .11, .12, .13, .14, .15, .16];
    const width = 2;
    const height = 2;
    const img = tf.tensor3d(data, [width, height, 4]);
    const canvas = getCanvas();

    tf.browser.draw(
        // tslint:disable-next-line:no-any
        img, canvas as any, {contextOptions: {contextType: env.name}});
    const actualData =
        await readPixelsFromGPUCanvas(canvas, height, width, unmultiplyAlpha);
    expectArraysClose(actualData, data, 0.01);
  });

  it('draw image with 3 channels and int values', async () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    const width = 2;
    const height = 2;
    const img = tf.tensor3d(data, [width, height, 3], 'int32');
    const canvas = getCanvas();

    tf.browser.draw(
        // tslint:disable-next-line:no-any
        img, canvas as any, {contextOptions: {contextType: env.name}});
    expectArraysEqual(
        await readPixelsFromGPUCanvas(canvas, height, width, removeLastChannel),
        data);
  });

  it('draw image with 3 channels and float values', async () => {
    const data = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .1, .11, .12];
    const width = 2;
    const height = 2;
    const img = tf.tensor3d(data, [width, height, 3]);
    const canvas = getCanvas();

    tf.browser.draw(
        // tslint:disable-next-line:no-any
        img, canvas as any, {contextOptions: {contextType: env.name}});
    const actualData =
        await readPixelsFromGPUCanvas(canvas, height, width, removeLastChannel);
    const expectedData = [25, 51, 76, 102, 128, 153, 178, 204, 229, 25, 28, 31];
    // On macOs/M2, some difference is close to 1.
    expectArraysClose(actualData, expectedData, 1);
  });

  it('draw 2D image in grayscale', async () => {
    const data = [1, 2, 3, 4];
    const width = 2;
    const height = 2;
    const img = tf.tensor2d(data, [width, height], 'int32');
    const canvas = getCanvas();

    tf.browser.draw(
        // tslint:disable-next-line:no-any
        img, canvas as any, {contextOptions: {contextType: env.name}});
    const actualData = await readPixelsFromGPUCanvas(canvas, height, width);
    const expectedData =
        [1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255, 4, 4, 4, 255];
    expectArraysEqual(actualData, expectedData);
  });

  it('draw image with alpha=0.5', async () => {
    const data = [101, 212, 113, 14, 35, 76, 17, 38, 59, 70, 81, 92];
    const width = 6;
    const height = 2;
    const img = tf.tensor3d(data, [width, height, 1], 'int32');
    const canvas = getCanvas();

    const drawOptions = {
      contextOptions: {contextType: env.name},
      imageOptions: {alpha: 0.5}
    };
    // tslint:disable-next-line:no-any
    tf.browser.draw(img, canvas as any, drawOptions);
    const actualData = await readPixelsFromGPUCanvas(canvas, height, width);

    const expectedData = [
      102, 102, 102, 128, 211, 211, 211, 128, 114, 114, 114, 128,
      14,  14,  14,  128, 36,  36,  36,  128, 76,  76,  76,  128,
      18,  18,  18,  128, 38,  38,  38,  128, 60,  60,  60,  128,
      70,  70,  70,  128, 82,  82,  82,  128, 92,  92,  92,  128
    ];
    expectArraysClose(actualData, expectedData, 2);
  });

  it('draw image works when canvas has been used for 2d', async () => {
    const data = [101, 212, 113, 14, 35, 76, 17, 38, 59, 70, 81, 92];
    const width = 6;
    const height = 2;
    const img = tf.tensor3d(data, [width, height, 1], 'int32');
    const canvas = getCanvas();
    // First use canvas as 2d.
    canvas.getContext('2d');

    const drawOptions = {
      contextOptions: {contextType: env.name},
      imageOptions: {alpha: 0.5}
    };
    // tslint:disable-next-line:no-any
    tf.browser.draw(img, canvas as any, drawOptions);
    const actualData = await readPixelsFromGPUCanvas(canvas, height, width);
    const expectedData = [
      102, 102, 102, 128, 211, 211, 211, 128, 114, 114, 114, 128,
      14,  14,  14,  128, 36,  36,  36,  128, 76,  76,  76,  128,
      18,  18,  18,  128, 38,  38,  38,  128, 60,  60,  60,  128,
      70,  70,  70,  128, 82,  82,  82,  128, 92,  92,  92,  128
    ];
    expectArraysClose(actualData, expectedData, 2);
  });
});
