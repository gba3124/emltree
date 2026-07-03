export type EMLNode =
  | { readonly t: '1' }
  | { readonly t: 'var'; readonly name: string }
  | { readonly t: 'eml'; readonly left: EMLNode; readonly right: EMLNode };

export interface Complex {
  re: number;
  im: number;
}

// core
export const ONE: EMLNode;
export function variable(name: string): EMLNode;
export function eml(left: EMLNode, right: EMLNode): EMLNode;
export function toNested(n: EMLNode): string;
export function toRpn(n: EMLNode): string;
export function leafCount(n: EMLNode): number;
export function nodeCount(n: EMLNode): number;
export function depth(n: EMLNode): number;
export function walk(n: EMLNode): Generator<EMLNode>;
export function asciiTree(n: EMLNode, prefix?: string, isLast?: boolean): string;

// constants
export const ZERO: EMLNode;
export const HALF: EMLNode;
export const E: EMLNode;
export const I: EMLNode;
export const PI: EMLNode;

// primitives
export function exp(x: EMLNode): EMLNode;
export function ln(x: EMLNode): EMLNode;
export function sub(x: EMLNode, y: EMLNode): EMLNode;
export function neg(x: EMLNode): EMLNode;
export function add(x: EMLNode, y: EMLNode): EMLNode;
export function mul(x: EMLNode, y: EMLNode): EMLNode;
export function div(x: EMLNode, y: EMLNode): EMLNode;
export function inv(x: EMLNode): EMLNode;
export function pow(x: EMLNode, y: EMLNode): EMLNode;
export function sqrt(x: EMLNode): EMLNode;
export function integer(n: number): EMLNode;
export function rational(p: number, q: number): EMLNode;
export function sinh(x: EMLNode): EMLNode;
export function cosh(x: EMLNode): EMLNode;
export function tanh(x: EMLNode): EMLNode;
export function asinh(x: EMLNode): EMLNode;
export function acosh(x: EMLNode): EMLNode;
export function atanh(x: EMLNode): EMLNode;
export function sin(x: EMLNode): EMLNode;
export function cos(x: EMLNode): EMLNode;
export function tan(x: EMLNode): EMLNode;
export function asin(x: EMLNode): EMLNode;
export function acos(x: EMLNode): EMLNode;
export function atan(x: EMLNode): EMLNode;
export function logBase(x: EMLNode, base: EMLNode): EMLNode;
export function sigmoid(x: EMLNode): EMLNode;

// compiler + evaluator
export function compile(formula: string): EMLNode;
export function evaluate(node: EMLNode, bindings?: Record<string, number | Complex>): Complex;
