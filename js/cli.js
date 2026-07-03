#!/usr/bin/env node
import {
  compile, toNested, toRpn, asciiTree, leafCount, nodeCount, depth, evaluate,
} from './index.js';

const args = process.argv.slice(2);
let formula;
let format = 'nested';
let stats = false;
let evalPairs = null;

for (let i = 0; i < args.length; i++) {
  const a = args[i];
  if (a === '-f' || a === '--format') format = args[++i];
  else if (a === '--stats') stats = true;
  else if (a === '--eval') evalPairs = args[++i] ?? '';
  else if (a === '-h' || a === '--help' || formula !== undefined) formula = undefined, i = args.length;
  else formula = a;
}

if (formula === undefined) {
  console.error(
    'usage: emltree "<formula>" [-f nested|rpn|tree] [--stats] [--eval x=0.9,y=2]\n'
    + 'Supports +, -, *, /, ^, **, sqrt, exp, log, ln, sin, cos, tan, asin, acos,\n'
    + 'atan, sinh, cosh, tanh, asinh, acosh, atanh, sigmoid, pi, E, I.',
  );
  process.exit(2);
}

let tree;
try {
  tree = compile(formula);
} catch (e) {
  console.error(`error: ${e.message}`);
  process.exit(2);
}

console.log(
  format === 'rpn' ? toRpn(tree)
  : format === 'tree' ? asciiTree(tree).trimEnd()
  : toNested(tree),
);

if (stats) {
  console.error(`[stats] leaves=${leafCount(tree)} nodes=${nodeCount(tree)} depth=${depth(tree)}`);
}

if (evalPairs !== null) {
  const bindings = {};
  for (const pair of evalPairs ? evalPairs.split(',') : []) {
    const [k, v] = pair.split('=');
    bindings[k.trim()] = Number(v);
  }
  const { re, im } = evaluate(tree, bindings);
  console.error(`[eval] ${re}${im < 0 ? '' : '+'}${im}i`);
}
