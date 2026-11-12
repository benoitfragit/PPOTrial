#include "flight_example.hpp"
#include "environment.hpp"
#include "environmentrenderer.hpp"
#include "serializer.hpp"
#include <iostream>
#include <string_view>

/** Dedicated desktop executable using the existing Environment and renderer. */
int main(int argc,char** argv)
{
    bool inference=false,check=false,autoRun=false;
    std::filesystem::path config=std::filesystem::path{RESOURCE_PATH}/"Scenarios/FlightControl3D.json";
    for(int i=1;i<argc;++i)
    {
        std::string_view const option(argv[i]);
        if(option=="--inference") inference=true;
        else if(option=="--check-example") check=true;
        else if(option=="--auto-run") autoRun=true;
        else if(option=="--config" && i+1<argc) config=argv[++i];
        else { std::cerr<<"Usage: FlightControl3D [--config FILE] [--inference] [--auto-run] [--check-example]\n"; return 2; }
    }
    auto const example=inference ? ppo::examples::flight::InferenceDefinition() : ppo::examples::flight::Definition();
    auto& serializer=ppo::serialization::Serializer::GetInstance();
    serializer.SetPath(std::filesystem::path{RESOURCE_PATH});
    ppo::model::Environment environment;
    if(!example.loadScenario(config,environment)) return 3;
    ppo::renderer::EnvironmentRenderer* renderer=nullptr;
    if(!check) renderer=&ppo::renderer::EnvironmentRenderer::GetInstance(std::string(example.windowTitle));
    serializer.Load();
    if(check) { std::cout<<"Bootstrapped dedicated FlightControl3D\n"; return 0; }
    renderer->SetEnvironment(environment,ppo::renderer::EnvironmentRenderer::View::FlightControl3D);
    environment.Start(); // Publish the initial preview while remaining paused.
    if(autoRun) environment.Play();
    while(renderer->Run()) {}
    environment.Stop();
    return 0;
}
