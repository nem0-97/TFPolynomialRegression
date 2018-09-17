let xs=[];
let ys=[];

//when drawing prediction shape how many steps to travel from -1 to 1 in
let res=200;

//not sure why this didn't work by putting height-mouseY into ys and mouseX into xs
//for some reason training would make slope and cept turn to NaN, same with mousY into ys and mouseX into xs

//equation degree and coefficients(stored in order of degree of x)
//eg. ax^2+bx+c is stored as coeffs=[c,b,a] (this makes guess function tiny bit easier)
let polyDegree=3;
let coeffs=[];

//optimizer and learning rate
let lR=.2;
let opt=tf.train.sgd(lR);

//function to guess
function guess(inds){
  let x=tf.tensor1d(inds);
  //this way create a 1dtensor with same length as x
  let result=x.pow(0).mul(coeffs[0]);
  //calculate a guess using coeffs
  for(let i=1;i<coeffs.length;i++){
    result=result.add(x.pow(i).mul(coeffs[i]));
  }
  return result;
}

//loss function to optimize(by minimizing)
function loss(gue,act){
  //return meanSquaredError
  return gue.sub(act).square().mean();
}

function setup(){
  createCanvas(innerWidth,innerHeight);
  //generate random coeffecients for a polynomial of degree polyDegree and put them into coeffs
  for(let i=0;i<=polyDegree;i++){
    coeffs.push(tf.scalar(random(1)).variable());
  }
}

function mousePressed(){
  xs.push(map(mouseX,0,width,-1,1));
  ys.push(map(mouseY,0,height,1,-1));
}

function draw(){
  tf.tidy(()=>{
    if(xs.length>0){
      //train automatically modifies only tensor variables no need to specify them
      opt.minimize(()=>loss(guess(xs),tf.tensor1d(ys)));
    }
  });

  background(0);

  //draw actual datapoints
  strokeWeight(16);
  stroke(124,165,57);
  for(let i=0;i<xs.length;i++){
    point(map(xs[i],-1,1,0,width),map(ys[i],-1,1,height,0));
  }

  //draw prediction equation
  let predXs=[];
  for(let i=-1.0;i<=1.0;i+=(2/res)){
    predXs.push(i);
  }
  let yVals;
  tf.tidy(()=>{
    //make a guess for y at edges of canvas then get data from tensor1d returned and use to draw line
    yVals=guess(predXs).dataSync();
  });
  //change to use dataSync instead of putting shape drawing in .then()
  //fixes shape not showing since data() is async and redrawing background
  strokeWeight(2);
  stroke(165,57,124);
  beginShape();
  noFill();
  for(let i=0;i<predXs.length;i++){
    vertex(map(predXs[i],-1,1,0,width),map(yVals[i],-1,1,height,0));
  }
  endShape();
}
