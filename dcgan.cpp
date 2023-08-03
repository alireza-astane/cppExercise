#include <torch/torch.h>
#include <ATen/ATen.h>
#include <iostream>

using namespace torch::indexing;

torch::Tensor J = torch::complex(torch::zeros(1),torch::ones(1));

class MolecularDynamic {
public:
    // the constructor to initialize the MD model
    MolecularDynamic(int N, float vMax, float sigma, float epsilon, int length);

    // some member variables
    int N; // the number of molecules
    float vMax; // the maximum speed of the molecules
    float sigma; // the distance param of the force
    at::Tensor p{};
    float epsilon; // the energy param of the force
    int length; // the size of the box
    float rc; // the threshold of the distance(we are gonna ignore further distances)
    torch::Tensor x; // the position of the molecules
    torch::Tensor v; // the velocity of the molecules
    torch::Tensor a; // the acceleration of the molecules
    torch::Tensor T; // the temperature of the system
    at::Tensor energyTrajectory;
    at::Tensor vTrajectory;
    at::Tensor xTrajectory;
    at::Tensor tempTrajectory;
    at::Tensor pressureTrajectory;
    at::Tensor uTrajectory;
    at::Tensor leftSideTrajectory;
    at::Tensor rMag;
    at::Tensor distances;
    at::Tensor filter;

    // some member functions
    torch::Tensor getK(); // get the kinetic energy
    torch::Tensor getU(); // get the potential energy
    torch::Tensor getAcceleration(); // get the acceleration from the force

    void run(float dt, int bigSteps, int smallSteps);

    void velocityVerlet(float dt);

    torch::Tensor getRebounded(torch::Tensor xNew);

    torch::Tensor U(torch::Tensor r);

    at::Tensor getLeftSidedNum();

    torch::Tensor getDistanceMatrix();

    torch::Tensor ljForce(torch::Tensor r);

    void save();
};
int main() {
    MolecularDynamic molecularDynamic = MolecularDynamic(100,20,1,1,50);
    molecularDynamic.run(0.00001,1000,1000);
    molecularDynamic.save();
}








void MolecularDynamic::save(){
    torch::save(energyTrajectory,"energyTrajectory.t");
    torch::save(vTrajectory,"vTrajectory.t");
    torch::save(xTrajectory,"xTrajectory.t");
    torch::save(tempTrajectory,"tempTrajectory.t");
    torch::save(pressureTrajectory,"pressureTrajectory.t");
    torch::save(uTrajectory,"uTrajectory.t");
    torch::save(leftSideTrajectory,"leftSideTrajectory.t");
}

void MolecularDynamic::run(float dt, int bigSteps, int smallSteps) {
    /*
    the function to go further in time for "steps" step of "dt"

    :param smallSteps: the big steps (getting data after each bigStep)
    :type smallSteps: int
    :param bigSteps: the small steps(total step of updating in bigStep)
    :type bigSteps:
    :param dt:the change of time in each step
    :type dt: float
    */

    //initializing the trajectories using PyTorch tensors and operations
    this->energyTrajectory = torch::zeros({bigSteps});
    this->xTrajectory = torch::zeros({bigSteps, this->N}, torch::kComplexDouble);
    this->vTrajectory = torch::zeros({bigSteps, this->N}, torch::kComplexDouble);
    this->tempTrajectory = torch::zeros({bigSteps});
    this->pressureTrajectory = torch::zeros({bigSteps});
    this->uTrajectory = torch::zeros({bigSteps});
    this->leftSideTrajectory = torch::zeros({bigSteps});

    //main loop
    for (int i = 0; i < bigSteps; i++) {
        std::cout << i << std::endl; //to see the process is running

        for (int j = 0; j < smallSteps; j++) {
            this->velocityVerlet(dt);
        }

        //storing data
        this->xTrajectory[i] = this->x;
        this->vTrajectory[i] = this->v;
        this->T = this->getK();
        this->tempTrajectory[i] = this->T;
        this->pressureTrajectory[i] = this->p;
        this->uTrajectory[i] = this->getU();
        this->energyTrajectory[i] = this->tempTrajectory[i] + this->uTrajectory[i];
        this->leftSideTrajectory[i] = this->getLeftSidedNum();
    }
}

torch::Tensor MolecularDynamic::getU() {
    // th function to calculate the total potential energy of the system
    // return: the total potential energy of the system
    torch::Tensor Us = U(rMag);
    Us.masked_fill_(torch::isnan(Us),0); // ignoring where U explodes for example U(rii)
    return torch::sum(Us*filter)/2;
}

at::Tensor MolecularDynamic::getLeftSidedNum() {


    return at::sum(torch::imag(x)<=length/2);
}

torch::Tensor MolecularDynamic::getK() {
    return torch::sum(torch::square(torch::real(v)) + torch::square(torch::imag(v)));
}

torch::Tensor MolecularDynamic::U(torch::Tensor r) {
    // the function to return the potential energy between to molecules based on lenard jones potential
    // param r:the distance between them
    // return: the potential energy
    return 4*epsilon*( torch::pow(sigma/r,12) - torch::pow(sigma/r,6));
}

torch::Tensor MolecularDynamic::getRebounded(torch::Tensor xNew) {
    // the function to rebound the positions using reminder of positions on length
    // param xNew: the unbounded positions
    // return: rebounded positions
    return (real(xNew) % length) + (imag(xNew) % length)*J;
}

torch::Tensor MolecularDynamic::getDistanceMatrix(){
    distances = torch::zeros({1,N,N});  //dtype
    torch::Tensor tiledPoses = torch::tile(x, {N, 1});
    distances = tiledPoses - torch::transpose(tiledPoses,0,1);
    distances = distances.reshape({N,N});
    distances = ((real(distances) + length/2) % length - length/2)  + J*((imag(distances) + length/2) % length - length/2);
    return distances;
}

torch::Tensor MolecularDynamic::ljForce(torch::Tensor r){
    rMag = torch::sqrt(torch::square(torch::real(r)) +  torch::square(torch::imag(r))) ;
    filter = (rMag < rc) ;
    torch::Tensor fMag = -24 * epsilon * (2 * torch::pow(sigma / rMag,14) - torch::pow(sigma / rMag,8) )* r ;
    fMag.masked_fill_(torch::isnan(fMag),0); // ignoring where U explodes for example U(rii)
    fMag *= filter;


    torch::Tensor fDotR = torch::sum(torch::real(fMag) * torch::real(r) + torch::imag(fMag) * torch::imag(r));
    p = (N*T + fDotR) / (2*length^2);
    return fMag;}

torch::Tensor MolecularDynamic::getAcceleration() {
    return at::sum(this->ljForce(this->getDistanceMatrix()),1);
}

MolecularDynamic::MolecularDynamic(int N, float vMax, float sigma, float epsilon, int length) {
    this->N = N;
    this->vMax = vMax;
    this->sigma = sigma;
    this->epsilon = epsilon;
    this->length = length;
    this->rc = 2.5 * sigma;

    int a = pow(2*N, 0.5);



    // creating available positions based on a using PyTorch tensors and operations
    auto X = torch::zeros({a * (a/2 + 1), 2});
    auto x1 = torch::linspace(0.1, 0.9, a+1);
    auto x2 = torch::linspace(0, 0.5, a/2);


    int index = 0;

    for (int i = 0; i < x1.size(0); i++) {
        for (int j = 0; j < x2.size(0); j++) {
            X[index][0] = x1[i];
            X[index][1] = x2[j];
            index++;
        }
    }


    this->x = this->length * (X.slice(0, 0, N).index({Slice(None),0}) +  X.slice(0, 0, N).index({Slice(None),1})*J); // choosing first N available positions
    // randomly initializing velocities using PyTorch tensors and operations
    this->v = torch::rand(N,torch::dtype(torch::kComplexFloat));
    this->v -= this->v.mean(0);
    this->v *= this->vMax;
    this->T = this->getK();
    this->a = this->getAcceleration();
}

void MolecularDynamic::velocityVerlet(float dt) {
    // the function to perform one update using velocity verlet algorithm
    // param dt: the change in time in the update
    torch::Tensor vNew = v + 0.5 * a * dt; // update velocities
    torch::Tensor xNew = (x + vNew * dt); // update positions
    x = this->getRebounded(xNew);

    torch::Tensor aNew = getAcceleration();
    vNew = vNew + 0.5 * aNew * dt; // update velocities

    v = vNew;
    v -= torch::mean(v); // resetting Vcm
    a = aNew;
}













