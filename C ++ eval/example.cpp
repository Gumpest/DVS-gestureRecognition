#include <torch/script.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <memory>
#include <ctime>
#include <map>

#include <torch/csrc/api/include/torch/utils.h>

#define ONLINE
//#define OFFLINE

#ifdef ONLINE
#include <mutex>
#include <opencv2/opencv.hpp>
#include <celex5/celex5.h>
#include <celex5/celex5datamanager.h>
#include <windows.h>
#define MAT_ROWS 800
#define MAT_COLS 1280
#define FPN_PATH   "E:/Dynamic Vision Sensor/CeleX5/Samples/Samples/config/FPN_2.txt"
#define MAX_BUFFER_NUM    5
#define USING_PROC_DATA_THREAD

CeleX5 *pCeleX5 = new CeleX5;
std::queue<std::vector<EventData>> g_queEventDataList;
CeleX5::CeleX5Mode   g_emSensorMode;
std::mutex           g_mtxDataLock;
std::vector<int> add_x, add_y, off_time;
int dx = 10;
int dy = 7;
int win = 30;

using namespace cv;
#endif 

const double CLOCKS_PER_SECOND = ((clock_t)1000);

using namespace std;

#ifdef ONLINE
class SensorDataObserver : public CeleX5DataManager
{
public:
	SensorDataObserver(CX5SensorDataServer* pServer)
	{
		m_pServer = pServer;
		m_pServer->registerData(this, CeleX5DataManager::CeleX_Frame_Data);
	}
	~SensorDataObserver()
	{
		m_pServer->unregisterData(this, CeleX5DataManager::CeleX_Frame_Data);
	}
	virtual void onFrameDataUpdated(CeleX5ProcessedData* pSensorData);//overrides Observer operation

	CX5SensorDataServer* m_pServer;
};

void SensorDataObserver::onFrameDataUpdated(CeleX5ProcessedData* pSensorData)
{
	if (NULL == pSensorData)
		return;
#ifdef USING_PROC_DATA_THREAD
	g_mtxDataLock.lock();
	std::vector<EventData> vecEvent;
	pCeleX5->getEventDataVector(vecEvent);
	g_queEventDataList.push(vecEvent);
	if (g_queEventDataList.size() > MAX_BUFFER_NUM )
	{
		g_queEventDataList.pop();
		//cout << "------ g_queEventDataList.size() > MAX_BUFFER_NUM ------" << endl;
	}
	g_mtxDataLock.unlock();
	g_emSensorMode = pSensorData->getSensorMode();
#else
	std::vector<EventData> vecEvent;
	if (CeleX5::Event_Off_Pixel_Timestamp_Mode == pSensorData->getSensorMode())
	{
		pCeleX5->getEventDataVector(vecEvent);
		cv::Mat mat = cv::Mat::zeros(cv::Size(1280, 800), CV_8UC1);
		int dataSize = vecEvent.size();
		for (int i = 0; i < dataSize; i++)
		{
			mat.at<uchar>(800 - vecEvent[i].row - 1, 1280 - vecEvent[i].col - 1) = 255;
		}
		if (dataSize > 0)
		{
			cv::imshow("Event Binary Pic", mat);
			cv::waitKey(1);
		}
	}
	else if (CeleX5::Event_In_Pixel_Timestamp_Mode == pSensorData->getSensorMode())
	{
		pCeleX5->getEventDataVector(vecEvent);
		cv::Mat mat = cv::Mat::zeros(cv::Size(1280, 800), CV_8UC1);
		int dataSize = vecEvent.size();
		for (int i = 0; i < dataSize; i++)
		{
			mat.at<uchar>(800 - vecEvent[i].row - 1, 1280 - vecEvent[i].col - 1) = 255;
		}
		if (dataSize > 0)
		{
			cv::imshow("Event Binary Pic", mat);
			cv::waitKey(1);
		}
	}
	else if (CeleX5::Event_Intensity_Mode == pSensorData->getSensorMode())
	{
		int count1 = 0, count2 = 0, count3 = 0;
		pCeleX5->getEventDataVector(vecEvent);
		cv::Mat mat = cv::Mat::zeros(cv::Size(1280, 800), CV_8UC1);
		int dataSize = vecEvent.size();
		for (int i = 0; i < dataSize; i++)
		{
			mat.at<uchar>(800 - vecEvent[i].row - 1, 1280 - vecEvent[i].col - 1) = (vecEvent[i].adc >> 4);
		}
		if (dataSize > 0)
		{
			//cout << "size = " << dataSize << ", t = " << vecEvent[dataSize - 1].t - vecEvent[0].t << endl;
			cv::imshow("Event Gray Pic", mat);
			cv::waitKey(1);
		}
	}
#endif
}
#endif




int main() {

#ifdef ONLINE
	
	map <int, string> res;
	res.insert(pair <int, string>(0, "arm roll"));
	res.insert(pair<int, string>(1, "arm updown"));
	res.insert(pair<int, string>(2, "hand clap"));
	res.insert(pair<int, string>(3, "right hand clockwise"));
	res.insert(pair<int, string>(4, "right hand wave"));

	torch::Tensor eventflow = torch::zeros({ 1, 1, 128, 128});
	string model_Path;
	model_Path = "D:/libtorch/example/gesture_ann.pt";							  //the location of  model

	torch::jit::script::Module module = torch::jit::load(model_Path);	      // load the model
	torch::NoGradGuard no_grad;
	module.eval();
	cout << "Model Loaded Done" << endl;

	if (NULL == pCeleX5)return 0;
	pCeleX5->openSensor(CeleX5::CeleX5_MIPI);
	pCeleX5->setFpnFile(FPN_PATH);
	pCeleX5->setSensorFixedMode(CeleX5::Event_Intensity_Mode);
	pCeleX5->disableFrameModule();
	pCeleX5->disableIMUModule();
	pCeleX5->disableEventCountSlice();
	pCeleX5->setEventFrameTime(30000);
	SensorDataObserver* pSensorData = new SensorDataObserver(pCeleX5->getSensorDataServer());

	while (true)
	{

		if (g_queEventDataList.size() == 0)
		{
			//cout << "----------- no data in vecEventList -----------" << endl;
		}
		else
		{
			g_mtxDataLock.lock();
			std::vector<EventData> vecEvent = g_queEventDataList.front();
			g_queEventDataList.pop();
			g_mtxDataLock.unlock();
			int dataSize = vecEvent.size();
			cv::Mat mat = cv::Mat::zeros(cv::Size(1280, 800), CV_8UC1);

			if (CeleX5::Event_Intensity_Mode == g_emSensorMode)
			{
				string ges ;
				for (int i = 1; i < dataSize; i++)
				{
					mat.at<uchar>(800 - vecEvent[i].row - 1, 1280 - vecEvent[i].col - 1) = 255;
					
					if (vecEvent[i].polarity != 0)			//存入数据流
					{
						int t = (vecEvent[i].tOffPixelIncreasing / 1000) % win;
						add_x.push_back(vecEvent[i].col / dx);
						add_y.push_back(vecEvent[i].row / dy);
						off_time.push_back(t);

						if ( (((vecEvent[i-1].tOffPixelIncreasing / 1000) % win) != 0) && t == 0 )
						{

						for (int j = 0; j < add_x.size(); j++) {

						eventflow[0][0][add_x[j]][add_y[j]] = 1;   //1 means an event

						}

						add_x.clear();
						add_y.clear();
						off_time.clear();

						vector<torch::jit::IValue> inputs;
						inputs.push_back(eventflow);

						torch::Tensor output = module.forward(std::move(inputs)).toTensor();

						auto max_result = output.max(1, true);
						auto max_index = std::get<1>(max_result).item<float>();
						ges = res[max_index];

						}

					}

					
											
				}

				vecEvent.clear();
								
				if (dataSize > 0)
				{
					putText(mat, ges, Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 0));
					cv::imshow("DVS_SNN_APPLICATION", mat);
					cv::waitKey(1);
				}				
			
			}

		}		

	}
	
	while (true)  {Sleep(1);}

	return 1;

#endif

#ifdef OFFLINE

	clock_t startTime, endTime;
	
	map <int, string> res;
	res.insert(pair <int, string>(0, "arm roll"));
	res.insert(pair<int, string>(1,"arm updown"));
	res.insert(pair<int, string>(2, "hand clap"));
	res.insert(pair<int, string>(3, "right hand clockwise"));
	res.insert(pair<int, string>(4, "right hand wave"));

	//csv_data off-line get the data
	/*
	vector<int> add_x, add_y, polarity, off_time;
	int dx = 10;
	int dy = 7;
	int dt = 30;
	int win = 2100;

	string file_path;
	file_path = "E:/gestureDemo_Python/data/2.csv";

	ifstream fin(file_path);                             // openfile
	string line;
	while (getline(fin, line))
	{
		istringstream sin(line);
		vector<string> Waypoints;
		string info;
		while (getline(sin, info, ','))
		{
			Waypoints.push_back(info);                   // a line into Waypoints
		}
		// Get x,y,p,t of a line and transform to int
		string x_str = Waypoints[1];
		string y_str = Waypoints[0];
		string p_str = Waypoints[3];
		string t_str = Waypoints[4];

		int x, y, p, t;
		stringstream sx, sy, sp, st;
		sx << x_str;
		sy << y_str;
		sp << p_str;
		st << t_str;

		sx >> x;
		sy >> y;
		sp >> p;
		st >> t;
		t = t / 1000;

		if (p == 0)continue;							// 0 means useless, so Not add the line
		if (t >= win)break;								// Need the data in the window
		add_x.push_back(x / dx);
		add_y.push_back(y / dy);
		if (p == -1)p = 0;
		polarity.push_back(p);
		off_time.push_back(t / dt);

	}
	//Data get done
	*/
	torch::Tensor eventflow = torch::zeros({ 1, 1, 128, 128});

//#pragma omp parallel for num_threads(7)

	//证明 Tensor可按索引赋值
	//eventflow[0][1][2][3][4] = 1;
	//cout << eventflow[0][1][2][3][4] << endl;
	
	//for (int i = 0; i < add_x.size(); i++) {
		//cout << add_x[i] << " " << add_y[i] << " " << polarity[i] << " " << off_time[i] << endl;

	//}
   /*
	for (int i = 0; i < add_x.size(); i++) {

		eventflow[0] [polarity[i]] [add_x[i]] [add_y[i]] [off_time[i]] = 1;   //1 means an event

	}*/

	string model_Path;
	model_Path = "D:/libtorch/example/gesture_ann.pt";							  //the location of  model

    
	torch::jit::script::Module module = torch::jit::load(model_Path);	      // load the model	
    torch::NoGradGuard no_grad;
	module.eval();
	cout << "Model Loaded Done" << endl;


	// Create a vector of inputs.
	vector<torch::jit::IValue> inputs;
	inputs.push_back(eventflow);

	// Execute the model and turn its output into a tensor.
	startTime = clock();
	torch::Tensor output = module.forward(inputs).toTensor();
	endTime = clock();

	auto max_result = output.max(1, true);
	auto max_index = std::get<1>(max_result).item<float>();
	cout << "The gesture is  " << res[max_index] << endl;
	cout << "Reference time is " << (double)(endTime - startTime) / CLOCKS_PER_SECOND << "s" << endl;

	system("pause");
	
#endif
}

