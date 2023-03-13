#include <cstdlib>
#include <iostream>
#include <fstream>
#include <ctime>
#include "kCurves.h"
#include <random>
using namespace std;

int main()
{	int control_size = 10;
	int size = 460;
	std::vector<Eigen::Vector2d> input(control_size);
	// input[0] << 4, 9.7;
	// input[1] << 0.1, -9.1;
	// input[2] << -6, -7.4;
	// input[3] << -6.7, 1.1;
    // input[4] << 7.3, 0.3;
	// input[5] << -7.9, -3.8;
	// input[6] << 3.2, 9.2;
	// input[7] << -9.6, -4.3;
	// input[8] << 2.5, -4.1;
	// input[9] << 7.7, -5.3;
	// input[0] << 2.1, 4.5;
	// input[1] << 1.9, 2.5;
	// input[2] << 2.3, 1.8;
	// input[3] << 2.7, 3.7;
    // input[4] << 2.9, 4.5;
	// input[5] << 3.3, 6.0;
	// input[6] << 3.9, 7.0;
	// input[7] << 2.5, 7.8;
	// input[8] << 1.2, 7.1;
	// input[9] << 1.6, 5.8;
	string path = "./training.txt";
    ofstream myfile (path);
	int i = 0;
	for (int j=0; j != size; ++j) 
	{
		for (int n=0; n != control_size; ++n) 
		{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dist9(0.0, 10.0); // distribution in range [-1, 1]
		input[n] << (std::floor(dist9(gen) * 10.) / 10), (std::floor(dist9(gen) * 10.) / 10);
		}
	std::vector<std::vector<Eigen::Vector2d>> bezierClosed = KCURVE::kCurveClosed(input);
	// std::cout << "Closed curve:" << std::endl;
    //     for (auto& seg : bezierClosed)
    //     {
    //             for (auto& pt : seg)
    //                     std::cout << pt << std::endl;
    //             std::cout << std::endl;
    //     }

	i = 0;
	// delete and redo any bezier with nan values
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
    // std::vector<std::vector<Eigen::Vector2d>> bezierOpen = KCURVE::kCurveOpen(input);
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