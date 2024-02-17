import std/[math, sequtils]


##########################################################################

proc roundToSignificantDigits(value: float, digits: int): float =
  if value == 0.0:
    return 0.0
  let d = ceil(log10(abs(value)))
  let power = digits - int(d)
  let magnitude = pow(10.0, power.float)
  round(value * magnitude) / magnitude


proc transpose(matrix: seq[seq[float]]): seq[seq[float]] =
  let numRows = matrix.len
  let numCols = matrix[0].len
  var transposed = newSeq[seq[float]](numCols)

  for i in 0 ..< numCols:
    transposed[i] = newSeq[float](numRows)
    for j in 0 ..< numRows:
      transposed[i][j] = matrix[j][i]

  return transposed


proc roundDown(n: float): float =
  ## values very close to zero are returned as zero
  const threshold: float = 1e-13
  if abs(n) < threshold:
    return 0.0
  return n


proc precomputeDCTCosineValues*(N: int): seq[seq[float]] =
  result = newSeqWith(N, newSeq[float](N))
  for k in 0 ..< N:
    for n in 0 ..< N:
      result[k][n] = cos(PI * k.float * (2.0 * n.float + 1.0) / (2.0 * N.float))


proc precomputeIDCTCosineValues*(N: int): seq[seq[float]] =
  result = newSeqWith(N, newSeq[float](N))
  for k in 0 ..< N:
    for n in 0 ..< N:
      result[k][n] = cos(PI * n.float * (2.0 * k.float + 1.0) / (2.0 * N.float))


proc precomputeDCTScalingFactors*(N: int): seq[float] =
  result = newSeq[float](N)
  result[0] = sqrt(1.0 / (4.0 * N.float)) * 2.0  # Special case for k=0
  for k in 1 ..< N:
    result[k] = sqrt(1.0 / (2.0 * N.float)) * 2.0  # General case for k>0


proc precomputeIDCTScalingFactors*(N: int): seq[float] =
  result = newSeq[float](N)
  for k in 0 ..< N:
    result[k] = sqrt(2.0 / N.float)

##########################################################################

proc dct1d*[T](vector: seq[T], ortho: bool = false): seq[float] =
  ## DCT II
  let N = vector.len.float
  result = newSeq[float](vector.len)

  for k in 0 ..< vector.len:
    var sum = 0.0
    for n in 0 ..< vector.len:
      sum += vector[n].float * cos(PI * k.float * (2.0 * n.float + 1.0) / (2.0 * N))
    
    if ortho:
      if k == 0:
        result[k] = roundDown((sqrt(1.0 / (4 * N)) * 2.0) * sum)
      else:
        result[k] = roundDown((sqrt(1.0 / (2 * N)) * 2.0) * sum)
    else:
      result[k] = roundDown(2.0 * sum)

  return result


proc dct1d*[T](vector: seq[seq[T]]): seq[seq[float]] =
  # DCT II (1D DCT over 2 Dimensional inputs)
  let N = vector.len
  result = newSeq[seq[float]](N)

  # NOTE: apply 1D DCT to each row
  for i in 0 ..< N:
    result[i] = dct1d(vector[i])

  return result


proc idct1d*[T](vector: seq[T], ortho: bool = false): seq[float] =
  ## DCT III
  let N = vector.len.float
  result = newSeq[float](vector.len)

  for k in 0 ..< vector.len:
    var sum = 0.0
    for n in 1 ..< vector.len:
      sum += vector[n] * cos(PI * n.float * (2.0 * k.float + 1.0) / (2.0 * N))
    
    if ortho:
      result[k] = roundDown((vector[0] / sqrt(N)) + (sqrt(2.0 / N)) * sum)
    else:
      result[k] = roundDown(vector[0] + (2.0 * sum))

  return result


proc idct1d*[T](vector: seq[seq[T]]): seq[seq[float]] =
  # DCT III (1D I-DCT over 2 Dimensional inputs)
  let N = vector.len
  result = newSeq[seq[float]](N)

  # NOTE: apply 1D IDCT to each row
  for i in 0 ..< N:
    result[i] = idct1d(vector[i])

  return result


proc dct2d*[T](matrix: seq[seq[T]], ortho: bool = false): seq[seq[float]] =
  ## DCT II
  let N = matrix.len
  var intermediate = newSeq[seq[float]](N)
  result = newSeq[seq[float]](N)

  # NOTE: apply 1D DCT to each row
  for i in 0 ..< N:
    intermediate[i] = dct1d(matrix[i], ortho)

  # NOTE: transpose the matrix
  let transposed = transpose(intermediate)

  # NOTE: apply 1D DCT to each transposed row (original column)
  for i in 0 ..< N:
    result[i] = dct1d(transposed[i], ortho)

  return transpose(result)


proc idct2d*[T](matrix: seq[seq[T]], ortho: bool = false): seq[seq[float]] =
  ## DCT III
  let N = matrix.len
  var intermediate = newSeq[seq[float]](N)
  result = newSeq[seq[float]](N)

  # NOTE: apply 1D IDCT to each row
  for i in 0 ..< N:
    intermediate[i] = idct1d(matrix[i], ortho)

  # NOTE: transpose the matrix
  let transposed = transpose(intermediate)

  # NOTE: apply 1D IDCT to each transposed row (original column)
  for i in 0 ..< N:
    result[i] = idct1d(transposed[i], ortho)

  # NOTE: transpose back to get the final result
  return transpose(result)

##########################################################################

when isMainModule:
  let input = @[
    @[1.0, 2.0, 3.0],
    @[4.0, 5.0, 6.0],
    @[7.0, 8.0, 9.0]
  ]

  let output = dct2d(input)
  echo output
  echo idct2d(output)
  
  let nOutput = dct2d(input, ortho=true)
  echo nOutput
  echo idct2d(nOutput, ortho=true)
