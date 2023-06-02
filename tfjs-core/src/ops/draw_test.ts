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

import * as tf from '../index';
import {BROWSER_ENVS, Constraints, describeWithFlags} from '../jasmine_util';
import {expectArraysClose, expectArraysEqual} from '../test_util';

class MockContext {
  data: ImageData;

  getImageData() {
    return this.data;
  }

  putImageData(data: ImageData, x: number, y: number) {
    this.data = data;
  }
}

class MockCanvas {
  context: MockContext;

  constructor(public width: number, public height: number) {}

  getContext(type: '2d'): MockContext {
    if (this.context == null) {
      this.context = new MockContext();
    }
    return this.context;
  }
}

const BROWSER_NO_WEBGPU_ENVS: Constraints = {
  predicate: (env) =>
      (env.backendName !== 'webgpu' && tf.env().platformName === 'browser')
};

describeWithFlags('Draw on 2d context', BROWSER_NO_WEBGPU_ENVS, () => {
  it('draw image with 4 channels and int values', async () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    const img = tf.tensor3d(data, [2, 2, 4], 'int32');
    const canvas = new MockCanvas(2, 2);
    const ctx = canvas.getContext('2d');

    // tslint:disable-next-line:no-any
    tf.browser.draw(img, canvas as any, {contextOptions: {contextType: '2d'}});
    expectArraysEqual(ctx.getImageData().data, data);
  });

  it('draw image with 4 channels and float values', async () => {
    const data =
        [.1, .2, .3, .4, .5, .6, .7, .8, .9, .1, .11, .12, .13, .14, .15, .16];
    const img = tf.tensor3d(data, [2, 2, 4]);
    const canvas = new MockCanvas(2, 2);
    const ctx = canvas.getContext('2d');

    // tslint:disable-next-line:no-any
    tf.browser.draw(img, canvas as any, {contextOptions: {contextType: '2d'}});
    const actualData = ctx.getImageData().data;
    const expectedData = data.map(e => Math.round(e * 255));
    expectArraysClose(actualData, expectedData, 1);
  });

  it('draw 2D image in grayscale', async () => {
    const data = [1, 2, 3, 4];
    const img = tf.tensor2d(data, [2, 2], 'int32');
    const canvas = new MockCanvas(2, 2);
    const ctx = canvas.getContext('2d');

    // tslint:disable-next-line:no-any
    tf.browser.draw(img, canvas as any, {contextOptions: {contextType: '2d'}});
    const actualData = ctx.getImageData().data;
    const expectedData =
        [1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255, 4, 4, 4, 255];
    expectArraysEqual(actualData, expectedData);
  });

  it('draw image with alpha=0.5', async () => {
    const data = [1, 2, 3, 4];
    const img = tf.tensor3d(data, [2, 2, 1], 'int32');
    const canvas = new MockCanvas(2, 2);
    const ctx = canvas.getContext('2d');

    const drawOptions = {
      contextOptions: {contextType: '2d'},
      imageOptions: {alpha: 0.5}
    };
    // tslint:disable-next-line:no-any
    tf.browser.draw(img, canvas as any, drawOptions);
    const actualData = ctx.getImageData().data;
    const expectedData =
        [1, 1, 1, 128, 2, 2, 2, 128, 3, 3, 3, 128, 4, 4, 4, 128];
    expectArraysEqual(actualData, expectedData);
  });
});

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

describeWithFlags('draw on webgpu context', BROWSER_ENVS, (env) => {
  let contextType: string;
  beforeAll(async () => {
    await tf.setBackend(env.name);
    contextType = env.name === 'cpu' ? '2d' : env.name;
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
        img, canvas as any, {contextOptions: {contextType}});
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
        img, canvas as any, {contextOptions: {contextType}});
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
        img, canvas as any, {contextOptions: {contextType}});
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
        img, canvas as any, {contextOptions: {contextType}});
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
        img, canvas as any, {contextOptions: {contextType}});
    const actualData = await readPixelsFromGPUCanvas(canvas, height, width);
    const expectedData =
        [1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255, 4, 4, 4, 255];
    expectArraysEqual(actualData, expectedData);
  });

  it('draw image with alpha=0.5', async () => {
    const data = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22];
    const width = 6;
    const height = 2;
    const img = tf.tensor3d(data, [width, height, 1], 'int32');
    const canvas = getCanvas();

    const drawOptions = {
      contextOptions: {contextType},
      imageOptions: {alpha: 0.5}
    };
    // tslint:disable-next-line:no-any
    tf.browser.draw(img, canvas as any, drawOptions);
    const actualData = await readPixelsFromGPUCanvas(canvas, height, width);
    const expectedData = [
      11, 11, 11, 255, 12, 12, 12, 255, 13, 13, 13, 255, 14, 14, 14, 255,
      15, 15, 15, 255, 16, 16, 16, 255, 17, 17, 17, 255, 18, 18, 18, 255,
      19, 19, 19, 255, 20, 20, 20, 255, 21, 21, 21, 255, 22, 22, 22, 255
    ];
    expectArraysClose(actualData, expectedData, 1);
  });

  it('draw image works when canvas has been used for 2d', async () => {
    const data = [11, 12, 13, 14];
    const width = 2;
    const height = 2;
    const img = tf.tensor3d(data, [width, height, 1], 'int32');
    const canvas = getCanvas();
    // First use canvas as 2d.
    canvas.getContext('2d');

    const drawOptions = {
      contextOptions: {contextType},
      imageOptions: {alpha: 0.5}
    };
    // tslint:disable-next-line:no-any
    tf.browser.draw(img, canvas as any, drawOptions);
    const actualData = await readPixelsFromGPUCanvas(canvas, height, width);
    const expectedData =
        [11, 11, 11, 255, 12, 12, 12, 255, 13, 13, 13, 255, 14, 14, 14, 255.];
    expectArraysClose(actualData, expectedData, 1);
  });
});
