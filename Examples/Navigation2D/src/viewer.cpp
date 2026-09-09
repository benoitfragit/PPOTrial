#include "navigation.hpp"
#include <raylib.h>
#include <algorithm>
#include <iostream>

namespace {
Vector2 screen(navigation::Point point,double extent)
{
    return {static_cast<float>(400+point.x/extent*350),static_cast<float>(400-point.y/extent*350)};
}
void draw(const navigation::Environment& environment,bool ended)
{
    const auto& scenario=environment.scenario();
    BeginDrawing();
    ClearBackground(RAYWHITE);
    DrawRectangleLines(50,50,700,700,DARKGRAY);
    for(const auto& obstacle:scenario.obstacles)
        DrawCircleV(screen(obstacle.center,scenario.half_extent),static_cast<float>(obstacle.radius/scenario.half_extent*350),GRAY);
    DrawCircleV(screen(scenario.goal,scenario.half_extent),
        std::max(3.F,static_cast<float>(scenario.arrival_radius/scenario.half_extent*350)),GREEN);
    DrawCircleV(screen(environment.position(),scenario.half_extent),5,BLUE);
    DrawText("Arrows: turn/accelerate | R: reset",15,10,20,DARKGRAY);
    if(ended) DrawText("Episode ended - press R",15,775,20,DARKGRAY);
    EndDrawing();
}
}
int main(int argc,char** argv)
{
    try
    {
        if(argc!=2) {std::cerr<<"Usage: navigation2d_viewer scenario.nav\n";return 2;}
        navigation::Environment environment(navigation::load_scenario(argv[1]));
        environment.reset();
        InitWindow(800,810,"Navigation2D");
        SetTargetFPS(60);
        bool ended=false;
        double accumulator=0;
        while(!WindowShouldClose())
        {
            if(IsKeyPressed(KEY_R)) {environment.reset();ended=false;accumulator=0;}
            accumulator=std::min(accumulator+GetFrameTime(),environment.scenario().dt+0.25);
            for(std::size_t substep=0;substep<16 && !ended && accumulator>=environment.scenario().dt;++substep)
            {
                const float turn=static_cast<float>(IsKeyDown(KEY_LEFT))-static_cast<float>(IsKeyDown(KEY_RIGHT));
                const float acceleration=static_cast<float>(IsKeyDown(KEY_UP))-static_cast<float>(IsKeyDown(KEY_DOWN));
                const auto result=environment.step(std::vector<float>{turn,acceleration});
                ended=result.terminated || result.truncated;
                accumulator-=environment.scenario().dt;
            }
            draw(environment,ended);
        }
        CloseWindow();
        return 0;
    }
    catch(const std::exception& error) {if(IsWindowReady()) CloseWindow();std::cerr<<error.what()<<'\n';return 1;}
}
