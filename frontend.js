"use strict";

const apiUrl = "";

document.addEventListener('DOMContentLoaded', init);

function init() {
  document.getElementById('submit_button').addEventListener('click', upload);
}

function upload(e) {
  e.preventDefault();

  let h = new Headers();
  h.append('Accept', 'application/json');

  let fd1 = new FormData();
  let fd2 = new FormData();

  let image1 = document.getElementById('image_1').files[0];
  let image2 = document.getElementById('image_2').files[0];

  fd1.append('image', image1)
  fd2.append('image', image2)

  let req1 = new Request(apiUrl, {
    method: 'POST',
    headers: h,
    body: fd1
  });

  let req2 = new Request(apiUrl, {
    method: 'POST',
    headers: h,
    body: fd2
  });

  let requests = [req1, req2].map( r => fetch(r).then(value => value.json()));

  Promise.all(requests)
    .then((values) => {
      setTimeout(() => {document.getElementById('results').innerHTML = 'Similarity: ' + cosineSimilarity(values[0],values[1])},0);
    })
}

function dotProduct(v1, v2) {
  if (v1.length != v2.length) {
    throw 'unequal vector lengths!';
  } else {
    let sum = 0;
    for (let i =0; i<v1.length; i++) {
      sum += v1[i] * v2[i]
    }
    return sum;
  }
}

function cosineSimilarity(v1, v2) {
  let norm_1 = dotProduct(v1, v1);
  let norm_2 = dotProduct(v2,v2);

  return dotProduct(v1, v2) / (norm_1*norm_2);
}
