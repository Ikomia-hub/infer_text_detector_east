#include "TextDetectorEAST.h"
#include "Graphics/CGraphicsLayer.h"

CTextDetectorEAST::CTextDetectorEAST() : COcvDnnProcess()
{
    m_pParam = std::make_shared<CTextDetectorEASTParam>();
    addOutput(std::make_shared<CGraphicsOutput>());
    addOutput(std::make_shared<CBlobMeasureIO>());
}

CTextDetectorEAST::CTextDetectorEAST(const std::string &name, const std::shared_ptr<CTextDetectorEASTParam> &pParam): COcvDnnProcess(name)
{
    m_pParam = std::make_shared<CTextDetectorEASTParam>(*pParam);
    addOutput(std::make_shared<CGraphicsOutput>());
    addOutput(std::make_shared<CBlobMeasureIO>());
}

size_t CTextDetectorEAST::getProgressSteps()
{
    return 3;
}

int CTextDetectorEAST::getNetworkInputSize() const
{
    return 576; //Multiple of 32
}

double CTextDetectorEAST::getNetworkInputScaleFactor() const
{
    return 1.0;
}

cv::Scalar CTextDetectorEAST::getNetworkInputMean() const
{
    return cv::Scalar(123.68, 116.78, 103.94);
}

void CTextDetectorEAST::run()
{
    beginTaskRun();
    auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
    auto pParam = std::dynamic_pointer_cast<CTextDetectorEASTParam>(m_pParam);

    if(pInput == nullptr || pParam == nullptr)
        throw CException(CoreExCode::INVALID_PARAMETER, "Invalid parameters", __func__, __FILE__, __LINE__);

    if(pInput->isDataAvailable() == false)
        throw CException(CoreExCode::INVALID_PARAMETER, "Empty image", __func__, __FILE__, __LINE__);

    //Force model files path
    pParam->m_modelFile = Utils::Plugin::getCppPath() + "/" + Utils::File::conformName(QString::fromStdString(m_name)).toStdString() + "/Model/east_text_detection.pb";

    CMat imgSrc;
    CMat imgOrigin = pInput->getImage();
    std::vector<cv::Mat> netOutputs;

    //Detection networks need color image as input
    if(imgOrigin.channels() < 3)
        cv::cvtColor(imgOrigin, imgSrc, cv::COLOR_GRAY2RGB);
    else
        imgSrc = imgOrigin;

    emit m_signalHandler->doProgress();

    try
    {
        if(m_net.empty() || pParam->m_bUpdate)
        {
            m_net = readDnn();
            if(m_net.empty())
                throw CException(CoreExCode::INVALID_PARAMETER, "Failed to load network", __func__, __FILE__, __LINE__);

            pParam->m_bUpdate = false;
        }

        int size = getNetworkInputSize();
        double scaleFactor = getNetworkInputScaleFactor();
        cv::Scalar mean = getNetworkInputMean();
        auto inputBlob = cv::dnn::blobFromImage(imgSrc, scaleFactor, cv::Size(size,size), mean, false, false);
        m_net.setInput(inputBlob);

        auto netOutNames = getOutputsNames();
        m_net.forward(netOutputs, netOutNames);
    }
    catch(cv::Exception& e)
    {
        throw CException(CoreExCode::INVALID_PARAMETER, e.what(), __func__, __FILE__, __LINE__);
    }

    readClassNames();
    endTaskRun();
    emit m_signalHandler->doProgress();
    manageOutput(netOutputs);
    emit m_signalHandler->doProgress();
}

void CTextDetectorEAST::manageOutput(const std::vector<cv::Mat>& netOutputs)
{
    forwardInputImage();

    if(netOutputs.size() < 2)
        throw CException(CoreExCode::INVALID_PARAMETER, "Wrong number of EAST Detector outputs", __func__, __FILE__, __LINE__);

    cv::Mat scores = netOutputs[1];
    cv::Mat geometry = netOutputs[0];
    CV_Assert(scores.dims == 4); CV_Assert(geometry.dims == 4); CV_Assert(scores.size[0] == 1);
    CV_Assert(geometry.size[0] == 1); CV_Assert(scores.size[1] == 1); CV_Assert(geometry.size[1] == 5);
    CV_Assert(scores.size[2] == geometry.size[2]); CV_Assert(scores.size[3] == geometry.size[3]);

    auto pParam = std::dynamic_pointer_cast<CTextDetectorEASTParam>(m_pParam);
    auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
    CMat imgSrc = pInput->getImage();

    std::vector<cv::RotatedRect> detections;
    std::vector<float> confidences;
    const int height = scores.size[2];
    const int width = scores.size[3];

    for (int y = 0; y < height; ++y)
    {
        const float* scoresData = scores.ptr<float>(0, 0, y);
        const float* x0_data = geometry.ptr<float>(0, 0, y);
        const float* x1_data = geometry.ptr<float>(0, 1, y);
        const float* x2_data = geometry.ptr<float>(0, 2, y);
        const float* x3_data = geometry.ptr<float>(0, 3, y);
        const float* anglesData = geometry.ptr<float>(0, 4, y);

        for (int x = 0; x < width; ++x)
        {
            float score = scoresData[x];
            if (score < pParam->m_confidence)
                continue;

            // Decode a prediction.
            // Multiple by 4 because feature maps are 4 time less than input image.
            float offsetX = x * 4.0f, offsetY = y * 4.0f;
            float angle = anglesData[x];
            float cosA = std::cos(angle);
            float sinA = std::sin(angle);
            float h = x0_data[x] + x2_data[x];
            float w = x1_data[x] + x3_data[x];

            cv::Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
            cv::Point2f p1 = cv::Point2f(-sinA * h, -cosA * h) + offset;
            cv::Point2f p3 = cv::Point2f(-cosA * w, sinA * w) + offset;
            cv::RotatedRect r(0.5f * (p1 + p3), cv::Size2f(w, h), -angle * 180.0f / (float)CV_PI);
            detections.push_back(r);
            confidences.push_back(score);
        }
    }

    // Apply non-maximum suppression procedure.
    std::vector<int> indices;
    cv::dnn::NMSBoxes(detections, confidences, pParam->m_confidence, pParam->m_nmsThreshold, indices);

    //Graphics output
    auto pGraphicsOutput = std::dynamic_pointer_cast<CGraphicsOutput>(getOutput(1));
    pGraphicsOutput->setNewLayer(getName());
    pGraphicsOutput->setImageIndex(0);

    //Measures output
    auto pMeasureOutput = std::dynamic_pointer_cast<CBlobMeasureIO>(getOutput(2));
    pMeasureOutput->clearData();

    int size = getNetworkInputSize();
    float xFactor = (float)imgSrc.cols / (float)size;
    float yFactor = (float)imgSrc.rows / (float)size;

    for(size_t i=0; i<indices.size(); ++i)
    {
        //Create polygon graphics of rotated box
        cv::RotatedRect& box = detections[indices[i]];
        cv::Point2f vertices[4];
        box.points(vertices);

        PolygonF poly;
        for(int j=0; j<4; ++j)
            poly.push_back(CPointF(vertices[j].x * xFactor, vertices[j].y * yFactor));

        auto graphicsPoly = pGraphicsOutput->addPolygon(poly);

        //Store values to be shown in results table
        std::vector<CObjectMeasure> results;
        results.emplace_back(CObjectMeasure(CMeasure(CMeasure::CUSTOM, QObject::tr("Confidence").toStdString()), confidences[indices[i]], graphicsPoly->getId(), "Text"));
        results.emplace_back(CObjectMeasure(CMeasure::Id::ORIENTED_BBOX, {box.center.x * xFactor, box.center.y * yFactor, box.size.width, box.size.height, box.angle}, graphicsPoly->getId(), "Text"));
        pMeasureOutput->addObjectMeasures(results);
    }
}

