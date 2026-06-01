/**
 * Ambient type declarations for `onnxruntime-node`.
 *
 * onnxruntime-node@1.19.2 ships no `.d.ts` files (its package `types` field
 * points at a file that isn't published), so TypeScript sees the module as
 * `any`. Rather than disable strictness, we declare the precise surface this
 * service uses. Mirrors the onnxruntime-common API.
 */
declare module "onnxruntime-node" {
  export type TensorDataType =
    | "float32"
    | "float64"
    | "int64"
    | "int32"
    | "uint8"
    | "int8"
    | "bool"
    | "string";

  export type TensorData =
    | Float32Array
    | Float64Array
    | BigInt64Array
    | Int32Array
    | Uint8Array
    | Int8Array;

  export class Tensor {
    constructor(type: TensorDataType, data: TensorData, dims: readonly number[]);
    readonly type: TensorDataType;
    readonly data: TensorData;
    readonly dims: readonly number[];
  }

  export interface SessionOptions {
    executionProviders?: string[];
    graphOptimizationLevel?: "disabled" | "basic" | "extended" | "all";
  }

  export interface InferenceSession {
    run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>>;
    readonly inputNames: readonly string[];
    readonly outputNames: readonly string[];
  }

  export namespace InferenceSession {
    function create(path: string, options?: SessionOptions): Promise<InferenceSession>;
    function create(
      buffer: Uint8Array | ArrayBuffer,
      options?: SessionOptions,
    ): Promise<InferenceSession>;
  }
}
