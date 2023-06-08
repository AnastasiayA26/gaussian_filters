#pragma once
#ifndef CTIKZ_HPP

#include <string>
#include <sstream>
#include <cmath>
#include <vector>

struct FunctionStyle{
    std::string color = "black";
    std::string mark = "none";
    std::string mark_size = "";
    bool only_marks = 0;
    int samples = 200;

    void write(std::stringstream& writing){
         writing << "\\addplot [" << std::endl;
        writing << "mark=" << mark << "," << std::endl;
        if(only_marks){
            writing << "only marks," << std::endl;
        }
        if(!mark_size.empty()){
            writing << "mark size=" << mark_size << "," << std::endl;
        }
        writing << "samples=" << samples << "," << std::endl;
        writing << "color=" << color << "]" << std::endl;
    }
};

struct FigureStyle{
    std::string color = "black";
    std::string patterned = "solid"; // default
    std::string arrow = "";
    std::string thickness = "";
    std::string fill = "none";

    void write(std::stringstream& writing){
        writing << "\\draw[";
        writing << "color=" << color << ", ";
        writing << "fill=" << fill << ", ";
        if (!thickness.empty()){
            writing << thickness << ", ";
        }
        if(!arrow.empty()){
            writing << arrow << ", ";
        }
        writing << patterned << "] ";
    }

};

struct AxisStyle{
    std::string title = "";
    std::string xlabel = "";
    std::string ylabel = "";

    std::string ymin = "";
    std::string xmin = "";
    std::string xmax = "";
    std::string ymax = "";

    std::string axis_lines = "box";
    bool grid = 0;
    std::string grid_style = "";
    double scale = 1;
    bool enlarge_limits = 1; 

    void write(std::stringstream& writing){
        if (!title.empty()){
            writing << "title=" << title << "," << std::endl;
        }
        if (!xlabel.empty()){
            writing << "xlabel=" << xlabel << "," << std::endl;
        }
        if (!ylabel.empty()){
            writing << "ylabel=" << ylabel << "," << std::endl;
        }       
        if(!xmin.empty()){
            writing << "xmin=" << xmin << "," << std::endl;
        }
        if(!ymin.empty()){
            writing << "ymin=" << ymin << "," << std::endl;
        }
        if(!xmax.empty()){
            writing << "xmax=" << xmax << "," << std::endl;
        }
        if(!ymax.empty()){
            writing << "ymax=" << ymax << "," << std::endl;
        }
        writing << "axis lines=" << axis_lines << "," << std::endl;
        if(grid){
            writing << "grid," << std::endl;
            if(!grid_style.empty()){
                writing << "grid style=" << grid_style << "," << std::endl;
            }
        }
        writing << "scale=" << scale << "," << std::endl;
        writing << "enlargelimits=" << std::boolalpha << enlarge_limits << "]" << std::endl;
    }
};

class CTikz{
public:
    CTikz();
    void create_tikz_file(std::string filename);
    void start_axis(AxisStyle style=AxisStyle()); // enlargelimits
    void end_axis();
    void drawFunc(std::vector<std::pair<double, double> > points, FunctionStyle style=FunctionStyle());
    void drawFunc(std::string func, FunctionStyle style=FunctionStyle());

    void start_picture();
    void end_picture();
    void drawCircle(double x1, double y1, double r, FigureStyle style=FigureStyle());
    void drawRectangle(double x1, double y1, double x2, double y2, FigureStyle style=FigureStyle());
    void drawGrid(double x1, double y1, double x2, double y2, double step=1, FigureStyle style=FigureStyle());
    void drawLines(std::vector<std::pair<double, double> > points, FigureStyle style=FigureStyle());

    void add_parse();

private:
    int block_status = 0; // 1 - axis, 2 - only tikz picture
    std::stringstream writing;
};

#endif //CTIKZ_HPP