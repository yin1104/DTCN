export const BenchmarkInfo = {
  targetNum: 40,
  _SUBJECTS: 35,
  _TEST_BLOCKS: [4, 5], // 0，1,2,3块为训练集和验证集，4,5用于测试集
  _TARGETS: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
    'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
    '_', ',', '.', '<'],
  _LOW_TARGETS: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
  'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
  'x', 'y', 'z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
  '_', ',', '.', '<'],
  _CHANNELS: [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2',
    'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
    'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'M1', 'TP7',
    'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5',
    'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4',
    'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2'
  ],
  _FREQS: [
    8, 9, 10, 11, 12, 13, 14, 15,
    8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2,
    8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4,
    8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6,
    8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8
  ],
  _PHASES: [
    0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5,
    0.5, 1, 1.5, 0, 0.5, 1, 1.5, 0,
    1, 1.5, 0, 0.5, 1, 1.5, 0, 0.5,
    1.5, 0, 0.5, 1, 1.5, 0, 0.5, 1,
    0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5
  ]
}

export const SSVEP_DATASET = [
  {
    name: 'Benchmark 40-class',
    info: BenchmarkInfo,
    value: 'benchmark'
  },
]