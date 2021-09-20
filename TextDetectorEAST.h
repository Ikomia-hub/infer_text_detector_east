#ifndef TEXTDETECTOREAST_H
#define TEXTDETECTOREAST_H

#include "TextDetectorEastGlobal.h"
#include "Process/OpenCV/dnn/COcvDnnProcess.h"
#include "Widget/OpenCV/dnn/COcvWidgetDnnCore.h"
#include "CPluginProcessInterface.hpp"

//----------------------------------//
//----- CTextDetectorEASTParam -----//
//----------------------------------//
class TEXTDETECTOREASTSHARED_EXPORT CTextDetectorEASTParam: public COcvDnnProcessParam
{
    public:

        CTextDetectorEASTParam() : COcvDnnProcessParam()
        {
            m_framework = Framework::TENSORFLOW;
        }

        void        setParamMap(const UMapString& paramMap) override
        {
            COcvDnnProcessParam::setParamMap(paramMap);
            m_confidence = std::stod(paramMap.at("confidence"));
            m_nmsThreshold = std::stod(paramMap.at("nmsThreshold"));
        }

        UMapString  getParamMap() const override
        {
            auto paramMap = COcvDnnProcessParam::getParamMap();
            paramMap.insert(std::make_pair("confidence", std::to_string(m_confidence)));
            paramMap.insert(std::make_pair("nmsThreshold", std::to_string(m_nmsThreshold)));
            return paramMap;
        }

    public:

        double m_confidence = 0.5;
        double m_nmsThreshold = 0.4;
};

//-----------------------------//
//----- CTextDetectorEAST -----//
//-----------------------------//
class TEXTDETECTOREASTSHARED_EXPORT CTextDetectorEAST: public COcvDnnProcess
{
    public:

        CTextDetectorEAST();
        CTextDetectorEAST(const std::string& name, const std::shared_ptr<CTextDetectorEASTParam>& pParam);

        size_t      getProgressSteps() override;
        int         getNetworkInputSize() const override;
        double      getNetworkInputScaleFactor() const override;
        cv::Scalar  getNetworkInputMean() const override;

        void        run() override;

    private:

        void        manageOutput(const std::vector<cv::Mat> &netOutputs);
};

//------------------------------------//
//----- CTextDetectorEASTFactory -----//
//------------------------------------//
class TEXTDETECTOREASTSHARED_EXPORT CTextDetectorEASTFactory : public CTaskFactory
{
    public:

        CTextDetectorEASTFactory()
        {
            m_info.m_name = "infer_text_detector_east";
            m_info.m_shortDescription = QObject::tr("Fast and accurate text detection in natural scenes using single neural network").toStdString();
            m_info.m_description = QObject::tr("Previous approaches for scene text detection have already achieved promising performances across various benchmarks. "
                                               "However, they usually fall short when dealing with challenging scenarios, even when equipped with deep neural network models, "
                                               "because the overall performance is determined by the interplay of multiple stages and components in the pipelines. "
                                               "In this work, we propose a simple yet powerful pipeline that yields fast and accurate text detection in natural scenes. "
                                               "The pipeline directly predicts words or text lines of arbitrary orientations and quadrilateral shapes in full images, "
                                               "eliminating unnecessary intermediate steps (e.g., candidate aggregation and word partitioning), with a single neural network. "
                                               "The simplicity of our pipeline allows concentrating efforts on designing loss functions and neural network architecture. "
                                               "Experiments on standard datasets including ICDAR 2015, COCO-Text and MSRA-TD500 demonstrate that "
                                               "the proposed algorithm significantly outperforms state-of-the-art methods in terms of both accuracy and efficiency. "
                                               "On the ICDAR 2015 dataset, the proposed algorithm achieves an F-score of 0.7820 at 13.2fps at 720p resolution.").toStdString();
            m_info.m_path = QObject::tr("Plugins/C++/Text/Detection").toStdString();
            m_info.m_version = "1.0.0";
            m_info.m_iconPath = "Icon/icon.png";
            m_info.m_authors = "Xinyu Zhou, Cong Yao, He Wen, Yuzhi Wang, Shuchang Zhou, Weiran He, Jiajun Liang";
            m_info.m_article = "EAST: An Efficient and Accurate Scene Text Detector";
            m_info.m_journal = "CVPR";
            m_info.m_year = 2017;
            m_info.m_license = "GPL 3.0 License";
            m_info.m_repo = "https://github.com/argman/EAST";
            m_info.m_keywords = "deep,learning,detection,tensorflow";
        }

        virtual WorkflowTaskPtr create(const WorkflowTaskParamPtr& pParam) override
        {
            auto paramPtr = std::dynamic_pointer_cast<CTextDetectorEASTParam>(pParam);
            if(paramPtr != nullptr)
                return std::make_shared<CTextDetectorEAST>(m_info.m_name, paramPtr);
            else
                return create();
        }
        virtual WorkflowTaskPtr create() override
        {
            auto paramPtr = std::make_shared<CTextDetectorEASTParam>();
            assert(paramPtr != nullptr);
            return std::make_shared<CTextDetectorEAST>(m_info.m_name, paramPtr);
        }
};

//-----------------------------------//
//----- CTextDetectorEASTWidget -----//
//-----------------------------------//
class TEXTDETECTOREASTSHARED_EXPORT CTextDetectorEASTWidget: public COcvWidgetDnnCore
{
    public:

        CTextDetectorEASTWidget(QWidget *parent = Q_NULLPTR): COcvWidgetDnnCore(parent)
        {
            init();
        }
        CTextDetectorEASTWidget(WorkflowTaskParamPtr pParam, QWidget *parent = Q_NULLPTR): COcvWidgetDnnCore(pParam, parent)
        {
            m_pParam = std::dynamic_pointer_cast<CTextDetectorEASTParam>(pParam);
            init();
        }

    private:

        void init()
        {
            if(m_pParam == nullptr)
                m_pParam = std::make_shared<CTextDetectorEASTParam>();

            auto pParam = std::dynamic_pointer_cast<CTextDetectorEASTParam>(m_pParam);
            assert(pParam);

            auto pSpinConfidence = addDoubleSpin(tr("Confidence"), pParam->m_confidence, 0.0, 1.0, 0.1, 2);
            auto pSpinNmsThreshold = addDoubleSpin(tr("NMS threshold"), pParam->m_nmsThreshold, 0.0, 1.0, 0.1, 2);
            
            //Connections
            connect(pSpinConfidence, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [&](double val)
            {
                auto pParam = std::dynamic_pointer_cast<CTextDetectorEASTParam>(m_pParam);
                assert(pParam);
                pParam->m_confidence = val;
            });
            connect(pSpinNmsThreshold, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [&](double val)
            {
                auto pParam = std::dynamic_pointer_cast<CTextDetectorEASTParam>(m_pParam);
                assert(pParam);
                pParam->m_nmsThreshold = val;
            });
        }

        void onApply() override
        {
            emit doApplyProcess(m_pParam);
        }
};

//------------------------------------------//
//----- CTextDetectorEASTWidgetFactory -----//
//------------------------------------------//
class TEXTDETECTOREASTSHARED_EXPORT CTextDetectorEASTWidgetFactory : public CWidgetFactory
{
    public:

        CTextDetectorEASTWidgetFactory()
        {
            m_name = "infer_text_detector_east";
        }

        virtual WorkflowTaskWidgetPtr   create(WorkflowTaskParamPtr pParam)
        {
            return std::make_shared<CTextDetectorEASTWidget>(pParam);
        }
};

//-----------------------------------//
//----- Global plugin interface -----//
//-----------------------------------//
class TEXTDETECTOREASTSHARED_EXPORT CTextDetectorEASTInterface : public QObject, public CPluginProcessInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "ikomia.plugin.process")
    Q_INTERFACES(CPluginProcessInterface)

    public:

        virtual std::shared_ptr<CTaskFactory> getProcessFactory()
        {
            return std::make_shared<CTextDetectorEASTFactory>();
        }

        virtual std::shared_ptr<CWidgetFactory> getWidgetFactory()
        {
            return std::make_shared<CTextDetectorEASTWidgetFactory>();
        }
};

#endif // TEXTDETECTOREAST_H
