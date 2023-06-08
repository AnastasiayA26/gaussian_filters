#include <ctikz/ctikz.hpp>
#include <fstream>

/*!
\brief Генерирует tikz файл
\param filename  название генерируемого файла
*/
void CTikz::create_tikz_file(std::string filename){
    if (block_status == 1){
        end_axis();
    }
    if (block_status == 2){
        end_picture();
    }
    block_status = 0;
    writing << "\\end{document}";
    filename += ".tex";
    std::ofstream file;
    file.open(filename);
    if (file.bad()){
        throw "File not created";
    }
    file << writing.str();
    return;
}

CTikz::CTikz(){
    writing.clear();
    writing << "\\documentclass[a4paper, 10pt]{article}" << std::endl;
    writing << "\\usepackage[margin=0.25in]{geometry}" << std::endl;
    writing << "\\usepackage{tikz, pgfplots}" << std::endl;
    writing << "\\pgfplotsset{compat=1.9}" << std::endl;
    writing << "\\begin{document}" << std::endl;

}
/*
\brief Конец блока с системой координат
*/
void CTikz::end_axis(){
    if (block_status == 1){
        writing << "\\end{axis}" << std::endl;
        writing << "\\end{tikzpicture}" << std::endl;
    }
    else {
        throw "Error";
    }
    block_status = 0;
}


// may be add title !!!!
void CTikz::start_axis(AxisStyle style){
    if(block_status != 0){
        throw "Error";
    }
    block_status = 1;
    writing << "\\begin{tikzpicture}" << std::endl;
    writing << "\\begin{axis}[" << std::endl;
    style.write(writing);

}

void CTikz::drawFunc(std::vector<std::pair<double, double>> points, FunctionStyle style){
    if(block_status == 0){
        throw "Axis not found";
    }   
    style.write(writing);
    writing << "coordinates{" << std::endl;
    for(auto & i: points){
        writing << "(" << i.first << "," << i.second << ")";
    }
    writing << std::endl << "};" << std::endl;
}

void CTikz::drawFunc(std::string func, FunctionStyle style){
    if(block_status != 1){
        throw "Axis not found";
    }
    style.write(writing);
    if (func.empty()){
        throw "Func is empty";
    }
    // check func !!!!!
    writing << "{" << func << "};" << std::endl;

}


void CTikz::start_picture(){
    if(block_status != 0){
        throw "Error";
    }
    writing << "\\begin{tikzpicture}" << std::endl;
    block_status = 2;
}

void CTikz::end_picture(){
    if(block_status == 2){
        writing << "\\end{tikzpicture}" << std::endl;
        block_status = 0;
    }
    else{
        throw "Eror";
    }
}

void CTikz::drawCircle(double x, double y, double r, FigureStyle style){
    if(block_status != 2){
        throw "Picture not found";
    }
    style.write(writing);
    writing << "(" << x << "," << y << ")" << " circle (" << r << ");" << std::endl;
}

void CTikz::drawRectangle(double x1, double y1, double x2, double y2, FigureStyle style){
    if(block_status != 2){
        throw "Picture not found";
    }
    style.write(writing);
    writing << "(" << x1 << "," << y1 << ") rectangle (" << x2 << "," << y2 << ");" << std::endl;
}

void CTikz::drawGrid(double x1, double y1, double x2, double y2, double step, FigureStyle style){
    if(block_status != 2){
        throw "Picture not found";
    }
    style.write(writing);
    writing << "(" << x1 << "," << y1 << ") grid " << "(" << x2 << "," << y2 << ");" << std::endl;
}

void CTikz::drawLines(std::vector<std::pair<double, double>> point, FigureStyle style){
    if(block_status != 2){
        throw "Picture not found";
    }
    style.write(writing);
    for(int i = 0; i < point.size() - 1; ++i){
        writing << "(" << point[i].first << "," << point[i].second << ") -- ";
    }
    writing << "(" << point[point.size() - 1].first << "," << point[point.size() - 1].second << ");" << std::endl;
    return;
}

void CTikz::add_parse(){
    if(block_status != 0){
        throw "Error! No blocks";
    }
    writing << std::endl;
}