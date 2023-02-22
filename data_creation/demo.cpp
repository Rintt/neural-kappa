#include <cstdlib>
#include <iostream>
#include <fstream>
#include <ctime>
#include "kCurves.h"
#include <random>
using namespace std;

int main()
{	int control_size = 3;
	int size = 10000;
	std::vector<Eigen::Vector2d> input(control_size);
	// input[0] << 3.1, 3.6;
	// input[1] << 4.3, 5.1;
	// input[2] << 6.9, 4.3;
	// input[3] << 9.5, 5.2;
    // input[4] << 9.75, 7.6;
	// input[5] << 7.9, 8.8;
	// input[6] << 5.8, 8;
	// input[7] << 4.4, 7.2;
	// input[8] << 2.8, 8.5;
	// input[9] << 3.3, 6.4;
	string path = "./training.txt";
    ofstream myfile (path);
	int i = 0;
	for (int j=0; j != size; ++j) 
	{
		for (int n=0; n != control_size; ++n) 
		{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dist9(-1.0, 1.0); // distribution in range [1, 6]
		input[n] << (std::floor(dist9(gen) * 10.) / 10), (std::floor(dist9(gen) * 10.) / 10);
		}
	std::vector<std::vector<Eigen::Vector2d>> bezierClosed = KCURVE::kCurveClosed(input);
	i = 0;
	//delete and redo any bezier with nan values
	if(isnan(bezierClosed[0][0][0])) {
		j = j-1;
		continue;
	}
	myfile << "Set of " << j << std::endl;

		for (auto& seg : bezierClosed)
		{
			myfile << "x: " << input[i][0] << " y: " << input[i][1];
			myfile << "\n";
				for (auto& pt : seg){
				myfile << pt << std::endl;
			}
			myfile << std::endl;
			i++;
		}
	}
    myfile.close();
	// std::cout << std::endl << "Open curve:" << std::endl;
	// for (auto& seg : bezierOpen)
	// {
	// 	for (auto& pt : seg)
	// 		std::cout << pt << std::endl;
	// 	std::cout << std::endl;
	// }
	std::cout << "New training set of " << size << " with " << control_size << " control points created at " << path << std::endl;
	return 0;
}